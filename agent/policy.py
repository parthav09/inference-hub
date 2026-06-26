class AgentPolicy:
    """Policy that defines when to use the agent and when to not"""

    def decide(self, value):
        if value.get("correctness_risk", 0) > 0.8:
            return "DISABLE_CACHE"
        
        if value.get("drift_score", 0) > 0.6:
            return "SHORT_TTL"
    
        return "NORMAL"
