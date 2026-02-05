import unittest
from src.agents.sentiment.sentiment_precise import SentimentPrecise
from src.agents.sentiment.sentiment_aggregator import _length_weighted
from src.agents.sentiment.chunker import chunk_text

class TestSentimentAgent(unittest.TestCase):

    def setUp(self):
        # Se ejecuta antes de cada test
        self.agent = SentimentPrecise()

    # --- 1. PRUEBAS DEL ROUTER (Idiomas) ---
    def test_router_english(self):
        """Verificar que texto en inglés va al modelo EN"""
        res = self.agent.analyze("I love this park in Quito", lang_hint="en")
        self.assertEqual(res["source"], "model_en_specialist")
        self.assertEqual(res["label"], "positive")

    def test_router_spanish(self):
        """Verificar que texto en español va al modelo ES"""
        res = self.agent.analyze("Odio el tráfico de la Av. Occidental", lang_hint="es")
        self.assertEqual(res["source"], "model_es_finetuned")
        self.assertEqual(res["label"], "negative")

    def test_router_default(self):
        """Verificar que sin hint o idioma raro, va al default (ES)"""
        res = self.agent.analyze("Hola mundo", lang_hint=None)
        self.assertEqual(res["source"], "model_es_finetuned")

    # --- 2. PRUEBAS DEL AGREGADOR (Matemáticas) ---
    def test_aggregation_logic(self):
        """
        Simulamos un post con 2 chunks:
        1. Largo (100 tokens) -> Positive (0.9)
        2. Corto (10 tokens)  -> Negative (0.6)
        El resultado debería ser POSITIVE porque el chunk largo pesa más.
        """
        fake_chunks = [
            {
                "sentiment": {"label": "positive", "confidence": 0.9},
                "span_tokens": [0, 100],
                "text": "long text..."
            },
            {
                "sentiment": {"label": "negative", "confidence": 0.6},
                "span_tokens": [100, 110], # solo 10 tokens de diferencia
                "text": "short..."
            }
        ]
        result = _length_weighted(fake_chunks)
        self.assertEqual(result["label_final"], "positive")

    # --- 3. PRUEBAS DEL CHUNKER ---
    def test_chunking_split(self):
        """Verificar que el chunker no rompa oraciones si caben"""
        text = "Hola. " * 50 # Texto repetitivo
        chunks = chunk_text(text, max_tokens=20, overlap=0)
        self.assertTrue(len(chunks) > 1)
        # Verificar que devuelve tuplas (texto, start, end)
        self.assertEqual(len(chunks[0]), 3) 

if __name__ == '__main__':
    unittest.main()