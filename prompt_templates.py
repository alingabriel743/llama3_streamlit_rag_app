memory_prompt_template ="""<s>[INST] Esti un chatbot care are o conversatie cu un utilizator. Ii vei raspunde la intrebari. Vei vorbi doar in limba romana, foarte clar,
    cu raspunsuri potrivite.[/INST]
    Conversatia anterioara este: {chat_history}
    Omul spune: {human_input}
    AI raspunde: """