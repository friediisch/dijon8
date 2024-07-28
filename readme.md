# Teil 1: Llama-3.1-8B-Agent
- Siehe code unter `src/main.py`.
- Der Code wurde auf einer Maschine entwickelt mit folgenden Eckdaten:
    - OS: Ubuntu 22.04
    - CPU: AMD 8-Core
    - GPU: RTX 3090
    - Python 3.12.3
  auf einem Endgerät mit weniger RAM/VRAM muss ggf. ein anderes Modell, bzw. Quantisation verwendet werden.
- Um Osrambot in Docker laufen zu lassen, verwende `docker build --tag=osrambot .` und `docker run -it osrambot`.
- Um den Code ohne Docker laufen zu lassen, führe die folgenden Schritte aus
    - Verwende `python3.12 -m venv venv` um eine neue virtuelle Umgebung zu erstellen (Python 3.12.4).
    - Activate it using `source venv/bin/activate`.
    - Install all dependencies using `pip install -r requirements-freeze.txt`.
    - Run Osrambot using `python src/main.py`
- Bei der ersten Verwendung müssen im ersten Schritt die Parameter für Llama-3.1-8B-Instruct heruntergeladen werden, sofern diese nicht bereits lokal durch Huggingface im Cache liegen.

# Teil 2: Verbesserungen in der Skalierung:

Ich würde erwarten dass das System auch mit 500 PDFs noch wie erwartet funktioniert. Alles darüber würde ggf. erfordern, das Retrieval der Chunks in einem eigenen Prozess auf dedizierter Infrastruktur laufen zu lassen und dafür entweder Provider-verwaltete Deployments (GCP Vertex Search) oder selbstverwaltete Open-Source Projekte wie zB. Qdrant zu wählen. Darüber hinaus ist auch das Retrieval über die Kosinusdistanz nicht ganz optimal, es gibt hier bessere, hierarchische Suchalgorithm, wie zB. Google ScaNN (https://github.com/google-research/google-research/tree/master/scann) oder HNSW (https://arxiv.org/abs/1603.09320). Mittels dieser Algorithmen hätte man auch kein Bottleneck im Retrieval zu erwarten. Dieses Setup hätte auch den Vorteil dass man Dokumente entkoppelt von der Bereitstellung des Modells an sich, wodurch man neue Dokumente hinzufügen und entfernen kann, ohne den Service offline nehmen zu müssen.

Ebenso kann diese Anwendung natürlich auch vom verwenden von LLM-APIs profitieren. Der Trade-Off bedeutet hier auf der einen Seite größere Auswahl der performantesten Closed-Source-Modelle, geringere Vorab-Anschaffungskosten für Infrastruktur, Fine-Tuning-as-a-Service und flexible laufende Kosten. Auf der anderen Seite ggf. höhere laufende Kosten, Einschränkungen bei der Verwendung sensibler Daten, erhöhte Latenz und eingeschränktere Möglichkeiten im Fine-Tuning durch die Nichtverfügbarkeit des Quellcodes und der Modellparameter.

Um das Grounding des Modells weiter zu verbessern und Fragen wie ‘Gebe mir alle Leuchtmittel mit mindestens 1500W und einer Lebensdauer von mehr als 3000 Stunden’ zu beantworten gibt es mehrere Möglichkeiten:
- Fine-Tuning: Durch das Fine-tunen mit passenden Daten und ggf. RLHF-Methoden kann das Grounding und Alignment des Modells verbessert werden.
- Propositions based splitting o.ä. (http://arxiv.org/pdf/2312.06648): Durch das generieren von passenden Chunks beim erstellen der Vektordatenbank kann man das Grounding des Modells verbessern.
- Pflege der Daten in einer tabellarischen Datenbank, in Kombination mit eine Retrieval-Pipeline welche die Nutzeraussage verwendet um die Datenbank z.B. mit SQL abzufragen und die Ergebnisse zu verwenden.
- Verwendung eines Kritiker-Modells: Indem man einen zweiten (dritten, vierten...) LLM-Aufruf verwendet um die Antwort, deren Relevanz, Korrektheit und Grounding źu bewerten und zu korrigieren, kann man die Antwortqualität weiter verbessern. Hier ist allerdings zu beachten dass dies Latenz und Compute-Kosten erhöht. 
- Nichtverwendung von Generativen Modellen: LLMs sind nicht immer die richtige Antwort, insbesondere in stark regulierten Industrien mit großen Reputationsrisiken. Hier können klassische Chatbots ggf. nach wie vor eine passende Alternative darstellen, auch wenn diese wesentlich weniger interessant und in ihren Antwortmöglichkeiten eingeschränkt sind...
 
# Teil 3: Evaluierungsmethoden:
Die Industrie befindet sich hier aktuell in einer Selbstfindungsphase. Es hat sich allerdings so weit die RAGAS (Retrieval Augmented Generation Assessment)-Methode bewährt (https://docs.ragas.io/en/stable/). Hierfür werden weitere LLM-Aufrufe verwendet, um eine gegebene Antwort auf z.B. Faithfulness, Relevancy, Context precision und Context recall zu bewerten. Dies kann ebenfalls helfen, die Answer correctness zu steigern und kann automatisiert passieren. Google Cloud Platform bietet außerdem LLM-Evaluationsmethoden wie zB. die "Rapid Evaluation": https://cloud.google.com/vertex-ai/generative-ai/docs/models/rapid-evaluation (currently in Pre-GA).


## Next steps:
Llama 3.1 unterstützt durch hugging face einen Modus, welcher Bilder mitverarbeiten kann und auf Document-QA-Tasks abzielt, dies könnte weitere Performance-Boni mit sich ziehen
- https://huggingface.co/docs/transformers/model_doc/layoutlmv3
- https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb


