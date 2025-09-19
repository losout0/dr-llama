def apply_disclaimer(answer: str) -> str:
    """
    Adiciona um disclaimer legal/informativo ao final da resposta.
    """
    disclaimer = (
        "\n\n---\n"
        "**Aviso Legal:** Eu sou um assistente de inteligência artificial "
        "e as minhas respostas são geradas com base em documentos públicos, "
        "tendo um caráter puramente informativo. **Este conteúdo não "
        "constitui e não substitui uma assessoria jurídica formal.** "
        "Sempre consulte um profissional qualificado para tratar de "
        "casos específicos."
    )
    
    return f"{answer}{disclaimer}"