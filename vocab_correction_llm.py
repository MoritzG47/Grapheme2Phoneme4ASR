"""
Plan:
Datenbank von Wörtern mit ihren passenden phonetischen Paaren.
KI darauf trainieren.
Wenn ein LLM ein komisches/unerwartetes Wort sieht, dann gibt es dieses Wort an die Phonetic-KI.
Diese gibt die nächsten Verwandten aus.
Das LLM entscheidet den besten Fit.
"""

from langchain_ollama import ChatOllama
SYS_PROMPT = """
            Korrigiere folgendes deutsches Transkript.
            Füge keine Informationen hinzu.
            Korrigiere die Grammatik.
            Korrigiere falsch transkribierte Wörter oder Sätze.
            Du bist in der folgenden Domäne: Computertechnik.
            Beachte folgende fachspezifische Ausdrücke:
            FHGenie -> internes LLM
            NER -> Named Entity Recognition
            Verändere den Text kontextbezogen und nur, wenn es sinnvoll ist.
            Gib mir am Ende nur zurück, welche Wörter komisch sind und im Gesamtkontext keinen Sinn ergeben.
            Transkribierter Text:
            """

llm = ChatOllama(
        model = "gemma4:e4b",               # Any available model on your local Ollama instance
        temperature=0,                      # Deterministic output
        format="json",                      # Force JSON output format
        base_url="http://localhost:11434",   # Standard Ollama port
        reasoning=True,
    )

transcript = "Lade das er Modell auf ein docke im Ätsch ich möchte auch dich auf potentiiele Probleme bei helfer jeans hinweisen"

response = llm.invoke(SYS_PROMPT + transcript)
print(response)