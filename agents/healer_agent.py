import time
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.model import SimpleNN, load_model_from_weights
from ai_engine.han_encryption import HANEncryption

class HealerAgent:
    def __init__(self):
        self.models = {}
        self.reputation_scores = {}
        self.encryption_handler = HANEncryption()
        self.global_model_weights = None
        self.healing_threshold = 0.5
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def register_model(self, node_id: str, weights: List[float], reputation: float = 100.0):
        try:
            self.models[node_id] = weights
            self.reputation_scores[node_id] = reputation
            
            self.logger.info(f"Model registered from node {node_id} with reputation {reputation}")
            
        except Exception as e:
            self.logger.error(f"Error registering model from node {node_id}: {e}")
    
    def detect_anomalous_models(self) -> List[str]:
        anomalous_nodes = []
        
        if len(self.models) < 2:
            return anomalous_nodes
        
        try:
            weight_vectors = list(self.models.values())
            
            mean_weights = np.mean(weight_vectors, axis=0)
            std_weights = np.std(weight_vectors, axis=0)
            
            for node_id, weights in self.models.items():
                weights_array = np.array(weights)
                
                if len(weights_array) != len(mean_weights):
                    min_len = min(len(weights_array), len(mean_weights))
                    weights_array = weights_array[:min_len]
                    mean_weights_trimmed = mean_weights[:min_len]
                    std_weights_trimmed = std_weights[:min_len]
                else:
                    mean_weights_trimmed = mean_weights
                    std_weights_trimmed = std_weights
                
                z_scores = np.abs((weights_array - mean_weights_trimmed) / (std_weights_trimmed + 1e-8))
                avg_z_score = np.mean(z_scores)
                
                if avg_z_score > 2.0:
                    anomalous_nodes.append(node_id)
                    self.logger.warning(f"Anomalous model detected from node {node_id} (Z-score: {avg_z_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalous models: {e}")
        
        return anomalous_nodes
    
    def heal_models(self, anomalous_nodes: List[str]) -> Dict[str, List[float]]:
        healed_models = {}
        
        try:
            for node_id in anomalous_nodes:
                if node_id in self.models:
                    original_weights = self.models[node_id]
                    
                    other_weights = [w for nid, w in self.models.items() if nid != node_id]
                    
                    if other_weights:
                        healed_weights = self._generate_healed_weights(original_weights, other_weights)
                        healed_models[node_id] = healed_weights
                        
                        self.models[node_id] = healed_weights
                        self.logger.info(f"Healed model for node {node_id}")
                    else:
                        self.logger.warning(f"Cannot heal model for node {node_id}: no reference models available")
                        
        except Exception as e:
            self.logger.error(f"Error healing models: {e}")
        
        return healed_models
    
    def _generate_healed_weights(self, anomalous_weights: List[float], reference_weights: List[List[float]]) -> List[float]:
        try:
            reference_array = np.array(reference_weights)
            mean_reference = np.mean(reference_array, axis=0)
            
            anomalous_array = np.array(anomalous_weights)
            
            min_len = min(len(anomalous_array), len(mean_reference))
            anomalous_trimmed = anomalous_array[:min_len]
            mean_reference_trimmed = mean_reference[:min_len]
            
            alpha = 0.7
            beta = 0.3
            
            healed_weights = alpha * mean_reference_trimmed + beta * anomalous_trimmed
            
            if len(healed_weights) < len(anomalous_weights):
                healed_weights = np.pad(healed_weights, (0, len(anomalous_weights) - len(healed_weights)), 'constant')
            elif len(healed_weights) > len(anomalous_weights):
                healed_weights = healed_weights[:len(anomalous_weights)]
            
            return healed_weights.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating healed weights: {e}")
            return anomalous_weights
    
    def aggregate_models(self, weight_by_reputation: bool = True) -> List[float]:
        try:
            if not self.models:
                return []
            
            if weight_by_reputation:
                total_reputation = sum(self.reputation_scores.values())
                if total_reputation == 0:
                    weights_array = np.array(list(self.models.values()))
                    return np.mean(weights_array, axis=0).tolist()
                
                weighted_weights = []
                for node_id, weights in self.models.items():
                    reputation = self.reputation_scores.get(node_id, 100.0)
                    weight_factor = reputation / total_reputation
                    weighted_weights.append(np.array(weights) * weight_factor)
                
                aggregated_weights = np.sum(weighted_weights, axis=0)
            else:
                weights_array = np.array(list(self.models.values()))
                aggregated_weights = np.mean(weights_array, axis=0)
            
            self.global_model_weights = aggregated_weights.tolist()
            return self.global_model_weights
            
        except Exception as e:
            self.logger.error(f"Error aggregating models: {e}")
            return []
    
    def update_reputation(self, node_id: str, performance_score: float):
        try:
            current_reputation = self.reputation_scores.get(node_id, 100.0)
            
            if performance_score > self.healing_threshold:
                new_reputation = min(100.0, current_reputation + 5.0)
            else:
                new_reputation = max(0.0, current_reputation - 10.0)
            
            self.reputation_scores[node_id] = new_reputation
            
            self.logger.info(f"Updated reputation for node {node_id}: {current_reputation} -> {new_reputation}")
            
        except Exception as e:
            self.logger.error(f"Error updating reputation for node {node_id}: {e}")
    
    def get_global_model(self) -> Optional[List[float]]:
        return self.global_model_weights
    
    def get_node_count(self) -> int:
        return len(self.models)
    
    def get_reputation_scores(self) -> Dict[str, float]:
        return self.reputation_scores.copy()
    
    def reset_models(self):
        self.models.clear()
        self.reputation_scores.clear()
        self.global_model_weights = None
        self.logger.info("Reset all models and reputation scores")

if __name__ == "__main__":
    healer = HealerAgent()
    
    print("Healer Agent initialized!")
    
    dummy_weights_1 = [1.0, 2.0, 3.0] * 100
    dummy_weights_2 = [1.1, 2.1, 3.1] * 100
    anomalous_weights = [10.0, 20.0, 30.0] * 100
    
    healer.register_model("node1", dummy_weights_1, 95.0)
    healer.register_model("node2", dummy_weights_2, 85.0)
    healer.register_model("node3", anomalous_weights, 75.0)
    
    anomalous = healer.detect_anomalous_models()
    print(f"Anomalous nodes detected: {anomalous}")
    
    if anomalous:
        healed = healer.heal_models(anomalous)
        print(f"Healed models for nodes: {list(healed.keys())}")
    
    global_model = healer.aggregate_models()
    print(f"Global model weights length: {len(global_model)}")
    
    print("Healer Agent test completed successfully!")
