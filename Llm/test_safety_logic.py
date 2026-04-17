import unittest
from advanced_highway_engine import SafetyExpertZakaria

class TestHighwayLogic(unittest.TestCase):
    def setUp(self):
        self.expert = SafetyExpertZakaria(speed_kmh=120, road_condition="sec")

    def test_legal_distance_dry(self):
        """Vérifie le calcul à 120 km/h sur route sèche"""
        #[span_1](end_span) Formule : (120/3.6) * 2 * 1.0 = 66.67m
        expected = round((120 / 3.6) * 2 * 1.0, 2)
        self.assertEqual(self.expert.calculate_legal_distance(), expected)

    def test_legal_distance_wet(self):
        """Vérifie le calcul sur route mouillée (coeff 1.5)"""
        [span_2](start_span)# La distance doit être augmentée de 50%
        self.expert.condition = "mouillé"
        expected = round((120 / 3.6) * 2 * 1.5, 2)
        self.assertEqual(self.expert.calculate_legal_distance(), expected)

    def test_triangulation_car(self):
        """Vérifie l'estimation de distance d'une voiture"""
        #[span_2](end_span) focal=1050, h_reelle_car=1.5m. Si h_px=150px -> d=10.5m
        dist = self.expert.triangulate_distance("car", 150)
        self.assertEqual(dist, 10.5)

if __name__ == '__main__':
    unittest.main()
