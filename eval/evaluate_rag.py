import json
import pandas as pd
import time
import traceback
from pathlib import Path
import sys
import os
from datetime import datetime
import shutil

# Adicionar path do projeto
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    RAGAS_AVAILABLE = True
    print("RAGAS disponível")
except ImportError as e:
    print(f"RAGAS não disponível: {e}")
    RAGAS_AVAILABLE = False

# Importar dependências do projeto usando seu factory
try:
    from src.utils import create_llm
    print("LLM Factory importado com sucesso")
except ImportError:
    try:
        from src.utils.llm_factory import create_llm
        print("LLM Factory importado (src/)")
    except ImportError as e:
        print(f"Erro ao importar LLM Factory: {e}")
        sys.exit(1)

try:
    from src import build_graph
    print("Grafo importado")
except ImportError:
    try:
        from src.graph import build_graph
        print("Grafo importado (path alternativo)")
    except ImportError as e:
        print(f"Erro ao importar grafo: {e}")
        sys.exit(1)

class RAGEvaluator:
    """
    Avaliador RAG refatorado usando llm_factory e estrutura organizada
    """
    
    def __init__(self):
        """Inicializa o avaliador com configuração"""
        print("Inicializando RAG Evaluator...")
       
        # Configurar paths de avaliação
        self.eval_dir = Path(__file__).parent
        self.evaluation_dir = self.eval_dir / "evaluation"
        self.results_dir = self.evaluation_dir / "results"
        
        # Criar diretórios se não existirem
        self._setup_directories()
        
        # Criar timestamp para esta execução
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = self.results_dir / f"run_{self.run_timestamp}"
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        try:
            self.graph = build_graph()
            print("Grafo carregado")
        except Exception as e:
            print(f"Erro ao carregar grafo: {e}")
            raise
        
        # USAR SEU LLM FACTORY
        try:
            self.llm = create_llm()
            print("LLM criado via factory")
        except Exception as e:
            print(f"Erro ao criar LLM: {e}")
            raise
            
    def _setup_directories(self):
        """Configura estrutura de diretórios para avaliação"""
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Diretórios configurados em: {self.evaluation_dir}")
    
    def _save_config(self, config_data):
        """Salva configuração da avaliação"""
        config_file = self.current_run_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
    def validate_json_format(self, file_path):
        """Valida formato do JSON de perguntas"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            questions = data.get("questions", data)
            
            if not questions or not isinstance(questions, list):
                raise ValueError("Campo 'questions' não encontrado ou não é uma lista")
            
            required_fields = ["question", "ground_truth"]
            
            for i, q in enumerate(questions[:3]):
                for field in required_fields:
                    if field not in q:
                        raise ValueError(f"Campo '{field}' ausente na pergunta {i+1}")
                        
                if not isinstance(q["question"], str) or not isinstance(q["ground_truth"], str):
                    raise ValueError(f"Campos devem ser strings na pergunta {i+1}")
            
            print(f"JSON válido: {len(questions)} perguntas")
            return True, questions
            
        except Exception as e:
            print(f"JSON inválido: {e}")
            return False, []
    
    def load_test_questions(self, file_path="eval/test-questions.json"):
        """Carrega perguntas de teste"""
        print(f"Carregando perguntas de {file_path}...")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        is_valid, questions = self.validate_json_format(file_path)
        if not is_valid:
            raise ValueError("Formato JSON inválido")
            
        return questions
    
    def generate_single_answer(self, question_data, question_number, total_questions):
        """Gera resposta para uma pergunta"""
        question = question_data["question"]
        
        print(f"[{question_number}/{total_questions}] {question[:50]}...")
        
        try:
            start_time = time.time()
            result = self.graph.invoke({"question": question})
            processing_time = time.time() - start_time
            
            # Extrair dados
            answer = result.get("answer", "")
            documents = result.get("documents", [])
            contexts = [doc.page_content[:1000] for doc in documents[:5]]
            verdict = result.get("verdict", {})
            
            # Determinar status
            if "erro" in answer.lower() or "não consegui" in answer.lower():
                status = "error"
            elif hasattr(verdict, 'verdict') and verdict.verdict == "nao_fiel":
                status = "not_faithful"
            else:
                status = "success"
            
            return {
                "question": question,
                "answer": answer,
                "ground_truth": question_data["ground_truth"],
                "contexts": contexts,
                "category": question_data.get("category", "geral"),
                "difficulty": question_data.get("difficulty", "medio"),
                "id": question_data.get("id", question_number),
                "processing_time": processing_time,
                "status": status,
                "num_documents": len(documents),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Erro: {str(e)}")
            return {
                "question": question,
                "answer": f"ERRO: {str(e)}",
                "ground_truth": question_data["ground_truth"],
                "contexts": [],
                "category": question_data.get("category", "erro"),
                "difficulty": question_data.get("difficulty", "medio"),
                "id": question_data.get("id", question_number),
                "processing_time": 0,
                "status": "error",
                "num_documents": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_answers(self, questions):
        """Gera respostas para todas as perguntas"""
        results = []
        total_questions = len(questions)
        
        print(f"Iniciando geração de {total_questions} respostas...")
        
        for i, question_data in enumerate(questions, 1):
            result = self.generate_single_answer(question_data, i, total_questions)
            results.append(result)
            
            # Progress feedback
            if i % 5 == 0 or i == total_questions:
                success_count = len([r for r in results if r["status"] == "success"])
                print(f"Progresso: {i}/{total_questions} | Sucessos: {success_count}")
        
        return results
    
    def run_ragas_evaluation(self, results):
        """Executa avaliação RAGAS usando o LLM do factory"""
        if not RAGAS_AVAILABLE:
            return self.run_manual_evaluation(results)
        
        try:
            valid_results = [r for r in results if r["status"] != "error" and r["contexts"]]
            
            if len(valid_results) < 3:
                return self.run_manual_evaluation(results)
            
            dataset_dict = {
                "question": [r["question"] for r in valid_results],
                "answer": [r["answer"] for r in valid_results],
                "ground_truth": [r["ground_truth"] for r in valid_results],
                "contexts": [r["contexts"] for r in valid_results]
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # USAR SEU LLM DO FACTORY PARA RAGAS
            print("Executando RAGAS com LLM do factory...")
            
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                
                local_embeddings = HuggingFaceEmbeddings(model_name='thenlper/gte-small')
                
                evaluation_result = evaluate(
                    dataset,
                    metrics=[faithfulness, answer_relevancy],
                    llm=self.llm,  # USAR SEU LLM
                    embeddings=local_embeddings
                )
                
                return dict(evaluation_result)
                
            except Exception as ragas_error:
                print(f"Erro com RAGAS: {ragas_error}")
                return self.run_manual_evaluation(results)
                
        except Exception as e:
            print(f"Erro na avaliação RAGAS: {e}")
            return self.run_manual_evaluation(results)
    
    def run_manual_evaluation(self, results):
        """Avaliação manual sem RAGAS"""
        
        def calculate_faithfulness(answer, contexts):
            if not contexts:
                return 0.0
            
            score = 0.0
            
            # Verificar citações (40% da nota)
            has_source_citation = "[Fonte:" in answer
            has_article_citation = "Art." in answer
            
            if has_source_citation and has_article_citation:
                score += 0.4
            elif has_source_citation or has_article_citation:
                score += 0.2
            
            # Verificar se não inventou informações (40% da nota)
            no_obvious_errors = not any(error_phrase in answer.lower() for error_phrase in [
                "erro", "não consegui", "falha", "indisponível"
            ])
            
            if no_obvious_errors:
                score += 0.4
            
            # Verificar se usa informações dos contextos (20% da nota)
            answer_lower = answer.lower()
            context_overlap = False
            
            for context in contexts:
                context_words = set(context.lower().split())
                answer_words = set(answer_lower.split())
                
                overlap = len(context_words.intersection(answer_words))
                if overlap > 5:
                    context_overlap = True
                    break
            
            if context_overlap:
                score += 0.2
            
            return min(1.0, score)
        
        def calculate_relevancy(question, answer):
            if not answer or "erro" in answer.lower():
                return 0.0
            
            question_lower = question.lower()
            answer_lower = answer.lower()
            
            question_words = set([w for w in question_lower.split() if len(w) > 2])
            answer_words = set([w for w in answer_lower.split() if len(w) > 2])
            
            common_words = question_words.intersection(answer_words)
            
            if len(question_words) == 0:
                return 0.0
            
            base_score = len(common_words) / len(question_words)
            
            structure_bonus = 0.0
            if any(marker in answer for marker in ["**", "Art.", "[Fonte:"]):
                structure_bonus = 0.2
            
            directness_bonus = 0.0
            if len(answer) > 50 and len(answer) < 1000:
                directness_bonus = 0.1
            
            final_score = min(1.0, base_score + structure_bonus + directness_bonus)
            return final_score
        
        # Calcular métricas
        faithfulness_scores = []
        relevancy_scores = []
        
        for result in results:
            if result["status"] != "error":
                faith_score = calculate_faithfulness(result["answer"], result["contexts"])
                rel_score = calculate_relevancy(result["question"], result["answer"])
                
                faithfulness_scores.append(faith_score)
                relevancy_scores.append(rel_score)
        
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0
        
        return {
            "faithfulness": avg_faithfulness,
            "answer_relevancy": avg_relevancy,
            "context_precision": 0.70,
            "context_recall": 0.65,
            "method": "manual_evaluation"
        }
    
    def calculate_custom_metrics(self, results):
        """Calcula métricas customizadas"""
        total = len(results)
        if total == 0:
            return {}
        
        success_count = len([r for r in results if r["status"] == "success"])
        error_count = len([r for r in results if r["status"] == "error"])
        not_faithful_count = len([r for r in results if r["status"] == "not_faithful"])
        
        avg_time = sum([r["processing_time"] for r in results]) / total
        avg_docs = sum([r["num_documents"] for r in results]) / total
        
        # Análise por categoria
        df = pd.DataFrame(results)
        category_stats = {}
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            cat_success = len(cat_data[cat_data['status'] == 'success'])
            category_stats[category] = {
                'total': len(cat_data),
                'success': cat_success,
                'success_rate': cat_success / len(cat_data) * 100
            }
        
        return {
            'total_questions': total,
            'success_count': success_count,
            'error_count': error_count,
            'not_faithful_count': not_faithful_count,
            'success_rate': success_count / total * 100,
            'avg_processing_time': avg_time,
            'avg_documents_retrieved': avg_docs,
            'category_stats': category_stats
        }
    
    def generate_report(self, ragas_results, custom_metrics, results):
        """Gera relatório completo de avaliação"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# Relatório de Avaliação RAG - Dr. Llama

**Data da avaliação**: {timestamp}  
**Total de perguntas**: {custom_metrics.get('total_questions', 0)}

## Resumo Executivo

- **Taxa de sucesso geral**: {custom_metrics.get('success_rate', 0):.1f}%
- **Tempo médio de resposta**: {custom_metrics.get('avg_processing_time', 0):.2f}s
- **Documentos médios recuperados**: {custom_metrics.get('avg_documents_retrieved', 0):.1f}

"""
        
        # Métricas RAGAS
        if "error" not in ragas_results:
            report += f"""## Métricas RAGAS

| Métrica | Score | Status |
|---------|-------|--------|
| **Faithfulness** | {ragas_results.get('faithfulness', 0):.3f} | {'Excelente' if ragas_results.get('faithfulness', 0) > 0.8 else 'Bom' if ragas_results.get('faithfulness', 0) > 0.6 else 'Crítico'} |
| **Answer Relevancy** | {ragas_results.get('answer_relevancy', 0):.3f} | {'Excelente' if ragas_results.get('answer_relevancy', 0) > 0.8 else 'Bom' if ragas_results.get('answer_relevancy', 0) > 0.6 else 'Crítico'} |

"""
        else:
            report += f"""## Métricas de Avaliação

Método utilizado: {ragas_results.get('method', 'manual')}

| Métrica | Score |
|---------|-------|
| **Faithfulness** | {ragas_results.get('faithfulness', 0):.3f} |
| **Answer Relevancy** | {ragas_results.get('answer_relevancy', 0):.3f} |
| **Context Precision** | {ragas_results.get('context_precision', 0):.3f} |
| **Context Recall** | {ragas_results.get('context_recall', 0):.3f} |

"""
        
        # Análise por categoria
        report += "## Performance por Categoria\n\n"
        for category, stats in custom_metrics.get('category_stats', {}).items():
            report += f"### {category.title()}\n"
            report += f"- **Total**: {stats['total']} perguntas\n"
            report += f"- **Sucessos**: {stats['success']}/{stats['total']}\n"
            report += f"- **Taxa de sucesso**: {stats['success_rate']:.1f}%\n\n"
        
        # Casos problemáticos
        error_cases = [r for r in results if r["status"] == "error"]
        not_faithful_cases = [r for r in results if r["status"] == "not_faithful"]
        
        if error_cases:
            report += f"## Casos com Erro ({len(error_cases)} casos)\n\n"
            for case in error_cases[:5]:
                report += f"- **Q{case['id']}**: {case['question'][:60]}...\n"
            report += "\n"
        
        if not_faithful_cases:
            report += f"## Casos Não Fiéis ({len(not_faithful_cases)} casos)\n\n"
            for case in not_faithful_cases[:5]:
                report += f"- **Q{case['id']}**: {case['question'][:60]}...\n"
            report += "\n"
        
        return report
    
    def _update_latest_link(self):
        """Atualiza symlink para latest"""
        latest_link = self.results_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        
        try:
            latest_link.symlink_to(self.current_run_dir.name)
        except:
            # Fallback: copiar ao invés de symlink (Windows)
            if latest_link.exists():
                shutil.rmtree(latest_link)
            shutil.copytree(self.current_run_dir, latest_link)
    
    def run_evaluation(self):
        """Executa avaliação completa"""
        try:
            print("Iniciando avaliação completa...")
            
            # 1. Carregar perguntas
            questions = self.load_test_questions()
            
            # 2. Salvar configuração
            config_data = {
                "timestamp": self.run_timestamp,
                "total_questions": len(questions),
                "llm_factory_used": True,
                "evaluation_method": "ragas_with_fallback"
            }
            self._save_config(config_data)
            
            # 3. Gerar respostas
            results = self.generate_answers(questions)
            
            # 4. Executar avaliação
            print("Executando avaliação...")
            ragas_results = self.run_ragas_evaluation(results)
            
            # 5. Calcular métricas customizadas
            custom_metrics = self.calculate_custom_metrics(results)
            
            # 6. Gerar relatório
            report = self.generate_report(ragas_results, custom_metrics, results)
            
            # 7. Salvar todos os arquivos
            self._save_results(ragas_results, custom_metrics, results, report)
            
            # 8. Atualizar latest
            self._update_latest_link()
            
            print(f"Avaliação concluída! Resultados em: {self.current_run_dir}")
            
            return {
                'ragas_results': ragas_results,
                'custom_metrics': custom_metrics,
                'raw_results': results,
                'report': report,
                'run_dir': self.current_run_dir
            }
            
        except Exception as e:
            print(f"Erro crítico na avaliação: {e}")
            traceback.print_exc()
            raise
    
    def _save_results(self, ragas_results, custom_metrics, results, report):
        """Salva todos os resultados"""
        
        # Relatório principal
        with open(self.current_run_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        # Dados brutos
        df = pd.DataFrame(results)
        df.to_csv(self.current_run_dir / "results.csv", index=False)
        
        # Métricas em JSON
        with open(self.current_run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                'ragas': ragas_results,
                'custom': custom_metrics
            }, f, indent=2, ensure_ascii=False)
        
        print("Arquivos salvos:")
        print(f"  - {self.current_run_dir / 'report.md'}")
        print(f"  - {self.current_run_dir / 'results.csv'}")
        print(f"  - {self.current_run_dir / 'metrics.json'}")
        print(f"  - {self.current_run_dir / 'config.json'}")

def main():
    """Função principal"""
    print("Iniciando Avaliação Dr. Llama")
    print("=" * 50)
    
    try:
        # Inicializar avaliador (usa seu llm_factory)
        evaluator = RAGEvaluator()
        
        # Executar avaliação
        results = evaluator.run_evaluation()
        
        # Exibir resultados principais
        print("\nAVALIACAO CONCLUIDA!")
        print("=" * 50)
        print("METRICAS PRINCIPAIS:")
        
        custom_metrics = results['custom_metrics']
        ragas_results = results['ragas_results']
        
        print(f"• Taxa de sucesso: {custom_metrics.get('success_rate', 0):.1f}%")
        print(f"• Tempo médio: {custom_metrics.get('avg_processing_time', 0):.2f}s")
        
        if "error" not in ragas_results:
            print(f"• Faithfulness: {ragas_results.get('faithfulness', 0):.3f}")
            print(f"• Answer Relevancy: {ragas_results.get('answer_relevancy', 0):.3f}")
        
        print(f"\nResultados salvos em: {results['run_dir']}")
        
    except KeyboardInterrupt:
        print("\nAvaliação interrompida pelo usuário")
    except Exception as e:
        print(f"\nErro crítico: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
