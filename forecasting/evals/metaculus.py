# forecasting/evals/metaculus.py
from inspect_ai import Task, task
from inspect_ai.scorer._forecast import peer_normalised_brier_score
from ..dataset.metaculus import MetaculusDataset, MetaculusConfig

@task
def metaculus_eval(config: MetaculusConfig = MetaculusConfig()) -> Task:
    """Creates a forecasting evaluation task using Metaculus questions"""
    return Task(
        dataset=MetaculusDataset(config),
        scorer=peer_normalised_brier_score(),
        description="Forecast the probability of future events using Metaculus questions"
    )

if __name__ == "__main__":
    import asyncio
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        config = MetaculusConfig(
            forecast_type=["binary"],
            tournaments=["q4-2024"],
            api_key="your-api-key-here"
        )
        
        task = metaculus_eval(config)
        async for sample in task.dataset:
            print(f"Question: {sample.input['title']}")
    
    asyncio.run(test())