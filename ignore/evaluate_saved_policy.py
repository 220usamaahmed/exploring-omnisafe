import os
import omnisafe

LOG_DIR = "/Users/usama/HRL/exploring-omnisafe/runs/DDPG-{SafetyPointGoal1-v0}/seed-000-2025-02-19-10-48-43"

if __name__ == "__main__":
    evaluator = omnisafe.Evaluator(render_mode="rgb_array")
    scan_dir = os.scandir(os.path.join(LOG_DIR, "torch_save"))
    for item in scan_dir:
        if item.is_file() and item.name.split(".")[-1] == "pt":
            evaluator.load_saved(
                save_dir=LOG_DIR,
                model_name=item.name,
                camera_name="track",
                width=256,
                height=256,
            )
            # evaluator.render(num_episodes=1)
            evaluator.evaluate(num_episodes=1)
    scan_dir.close()
