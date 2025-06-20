import unittest
import numpy as np
import sys
import os

# This setup tries to ensure that the code can be run directly
# or via a test runner like 'python -m unittest discover' from the project root.
# It adds the parent directory of 'Chapter3' to the Python path.
current_script_path = os.path.abspath(__file__)
chapter3_dir = os.path.dirname(current_script_path) # e.g., /path/to/RL_Cookbook_Codes_MJ/Chapter3
project_root_dir = os.path.dirname(chapter3_dir) # e.g., /path/to/RL_Cookbook_Codes_MJ
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

# Now, import modules using their path from the project root.
# The environment file is '01_robot_warehouse_env.py'
from Chapter3.01_robot_warehouse_env import OrderPickingEnv
from Chapter3.common import patch_plt_show, unpatch_plt_show

class TestOrderPickingEnv(unittest.TestCase):
    def setUp(self):
        patch_plt_show()
        self.grid_size = (7, 7)
        self.depot_pos = (0, 0)
        self.item_locations = [(1, 1), (3, 3)]
        self.obstacle_locations = [(2, 2)]

        self.env = OrderPickingEnv(
            grid_size=self.grid_size,
            depot_pos=self.depot_pos,
            item_locations=self.item_locations,
            obstacle_locations=self.obstacle_locations,
            max_steps_per_episode=50
        )
        self.env.reset()

    def tearDown(self):
        if hasattr(self.env, 'close') and callable(self.env.close):
            self.env.close()
        unpatch_plt_show()

    def test_pickup_item(self):
        # Override item locations for this specific test
        self.env.item_locations = [np.array([0, 1])] # Item at (0,1)
        self.env.num_items = len(self.env.item_locations)
        self.env.obstacle_locations = [] # No obstacles for simplicity
        self.env.reset() # Apply new item/obstacle configuration

        # Robot starts at depot (0,0)
        # Action 3 is 'Right'
        # Expected movement_reward = -0.1
        # Expected item_pickup_reward = 15.0
        # Expected total_reward = 14.9

        # Initial state check
        initial_state = self.env._get_state()
        self.assertEqual(initial_state['robot_pos'], (0,0))
        self.assertFalse(initial_state['items_picked'][0])

        action = 3 # Move Right from (0,0) to (0,1)
        new_state, reward, done, info = self.env.step(action)

        self.assertEqual(new_state['robot_pos'], (0, 1), "Robot should move to item location")
        self.assertTrue(new_state['items_picked'][0], "Item should be marked as picked")
        self.assertAlmostEqual(reward, 15.0 - 0.1, delta=0.001, msg="Reward should be item pickup bonus minus movement cost")

        # Since it's the only item, all_items_collected_phase should be True
        self.assertTrue(new_state['all_items_collected_phase'], "Should be in return to depot phase")
        self.assertFalse(done, "Game should not be done yet, robot needs to return to depot")
        self.assertEqual(info['items_left_to_pick'], 0)
        self.assertTrue(info['in_return_to_depot_phase'])

    def test_hit_obstacle(self):
        # Obstacle at (0,1), robot at (0,0)
        self.env.obstacle_locations = [np.array([0, 1])]
        self.env.item_locations = [np.array([1,1])] # Item elsewhere
        self.env.num_items = len(self.env.item_locations)
        self.env.reset()

        initial_robot_pos_tuple = tuple(self.env.robot_pos) # robot_pos is np.array, convert for state comparison
        initial_state_from_env = self.env._get_state()
        initial_items_picked_tuple = initial_state_from_env['items_picked']


        # Action 3 is 'Right'. Attempt to move from (0,0) into obstacle at (0,1)
        # Expected movement_reward = -0.7 (for hitting obstacle)
        # No item pickup reward, no completion reward.
        action = 3
        new_state, reward, done, info = self.env.step(action)

        self.assertEqual(new_state['robot_pos'], initial_robot_pos_tuple,
                        f"Robot position should not change after hitting obstacle. Expected {initial_robot_pos_tuple}, got {new_state['robot_pos']}")
        self.assertAlmostEqual(reward, -0.7, delta=0.001,
                               msg=f"Reward should be penalty for hitting obstacle. Expected -0.7, got {reward}")
        self.assertFalse(done, "Game should not be done after hitting an obstacle")
        self.assertEqual(new_state['items_picked'], initial_items_picked_tuple,
                         "Items picked status should not change after hitting obstacle")
        self.assertFalse(new_state['all_items_collected_phase'],
                         "Should not be in return to depot phase after hitting obstacle early")

    def test_hit_wall(self):
        # Environment uses default grid_size=(7,7) and depot_pos=(0,0)
        # Robot starts at (0,0)
        self.env.item_locations = [np.array([1,1])] # Item elsewhere
        self.env.num_items = len(self.env.item_locations)
        self.env.obstacle_locations = [] # No obstacles
        self.env.reset()

        initial_robot_pos_tuple = self.env._get_state()['robot_pos'] # Should be (0,0)
        initial_items_picked_tuple = self.env._get_state()['items_picked']

        # Action 0 is 'Up'. Attempt to move from (0,0) Up, hitting the wall.
        # Expected movement_reward = -0.5 (for hitting wall)
        action = 0
        new_state, reward, done, info = self.env.step(action)

        self.assertEqual(new_state['robot_pos'], initial_robot_pos_tuple,
                        f"Robot position should not change after hitting wall. Expected {initial_robot_pos_tuple}, got {new_state['robot_pos']}")
        self.assertAlmostEqual(reward, -0.5, delta=0.001,
                               msg=f"Reward should be penalty for hitting wall. Expected -0.5, got {reward}")
        self.assertFalse(done, "Game should not be done after hitting a wall")
        self.assertEqual(new_state['items_picked'], initial_items_picked_tuple,
                         "Items picked status should not change after hitting wall")
        self.assertFalse(new_state['all_items_collected_phase'],
                         "Should not be in return to depot phase after hitting wall early")

        # Try hitting another wall: Action 2 (Left) from (0,0)
        self.env.reset() # Reset robot to (0,0)
        initial_robot_pos_tuple = self.env._get_state()['robot_pos']
        initial_items_picked_tuple = self.env._get_state()['items_picked']

        action = 2 # Move Left
        new_state, reward, done, info = self.env.step(action)
        self.assertEqual(new_state['robot_pos'], initial_robot_pos_tuple,
                        f"Robot position should not change after hitting wall (left). Expected {initial_robot_pos_tuple}, got {new_state['robot_pos']}")
        self.assertAlmostEqual(reward, -0.5, delta=0.001,
                               msg=f"Reward should be penalty for hitting wall (left). Expected -0.5, got {reward}")

    def test_complete_task_pickup_and_return(self):
        # Depot at (0,0), Item at (0,1)
        self.env.item_locations = [np.array([0, 1])]
        self.env.num_items = len(self.env.item_locations)
        self.env.obstacle_locations = [] # No obstacles
        self.env.reset()

        # Sanity check initial state
        initial_state = self.env._get_state()
        self.assertEqual(initial_state['robot_pos'], (0,0), "Robot should start at depot")
        self.assertFalse(initial_state['items_picked'][0], "Item should start unpicked")

        # Step 1: Move Right from (0,0) to (0,1) to pick item
        # Action 3: Right
        # Expected reward: -0.1 (move) + 15.0 (pickup) = 14.9
        action_pickup = 3
        state_after_pickup, reward_pickup, done_pickup, info_pickup = self.env.step(action_pickup)

        self.assertEqual(state_after_pickup['robot_pos'], (0,1), "Robot should be at item location")
        self.assertTrue(state_after_pickup['items_picked'][0], "Item should be picked")
        self.assertAlmostEqual(reward_pickup, 15.0 - 0.1, delta=0.001, "Reward for pickup is incorrect")
        self.assertTrue(state_after_pickup['all_items_collected_phase'], "Should be in return to depot phase")
        self.assertFalse(done_pickup, "Task should not be done yet, need to return to depot")
        self.assertEqual(info_pickup['items_left_to_pick'], 0)
        self.assertTrue(info_pickup['in_return_to_depot_phase'])

        # Step 2: Move Left from (0,1) to (0,0) to return to depot
        # Action 2: Left
        # Expected reward: -0.1 (move) + 100.0 (completion) = 99.9
        action_return = 2
        final_state, reward_return, done_return, info_return = self.env.step(action_return)

        self.assertEqual(final_state['robot_pos'], (0,0), "Robot should be back at depot")
        self.assertTrue(final_state['items_picked'][0], "Item should still be picked") # Ensure item status persists
        self.assertAlmostEqual(reward_return, 100.0 - 0.1, delta=0.001, "Reward for return and completion is incorrect")
        self.assertTrue(done_return, "Task should be done after returning to depot")
        self.assertTrue(final_state['all_items_collected_phase'], "Should remain in return to depot phase")
        # items_left_to_pick might be 0 or not present in info if already in return phase, check env spec
        # For this env, info['items_left_to_pick'] is 0 once in return phase.
        self.assertEqual(info_return['items_left_to_pick'], 0)
        self.assertTrue(info_return['in_return_to_depot_phase'])

    def test_episode_ends_due_to_max_steps(self):
        # Configure env with a very short episode length
        # Item far away, or simply perform non-completing actions
        custom_item_locs = [(3,3)] # Item that won't be picked quickly

        # Re-initialize environment with specific max_steps
        # Need to close the old one if necessary, or ensure setUp handles it.
        # self.env.close() # Assuming close is idempotent or setUp handles it

        # Create a new env instance for this test with different max_steps
        # This is safer than modifying self.env if other tests depend on its original max_steps
        # However, for unittest structure, usually we modify self.env in setUp or the test itself.
        # Let's modify self.env.max_steps and then reset.

        # original_max_steps = self.env.max_steps # Save original if needed, though setUp resets
        self.env.max_steps_per_episode = 2 # Environment uses max_steps_per_episode
        self.env.item_locations = [np.array(loc) for loc in custom_item_locs]
        self.env.num_items = len(self.env.item_locations)
        self.env.obstacle_locations = []
        self.env.reset() # Reset with new max_steps and item configuration

        # Sanity check: Robot at (0,0), item at (3,3), max_steps = 2
        self.assertEqual(self.env._get_state()['robot_pos'], (0,0))
        self.assertFalse(self.env._get_state()['items_picked'][0])
        self.assertEqual(self.env.max_steps_per_episode, 2)
        self.assertEqual(self.env.current_step, 0)

        # Action 1: Move Down (Action 1) from (0,0) to (1,0)
        # Reward: -0.1. Done: False. Current_step: 1
        _, reward1, done1, _ = self.env.step(1) # Move Down
        self.assertAlmostEqual(reward1, -0.1, delta=0.001)
        self.assertFalse(done1)
        self.assertEqual(self.env.current_step, 1)
        self.assertFalse(self.env._get_state()['all_items_collected_phase']) # Item not picked

        # Action 2: Move Down (Action 1) from (1,0) to (2,0)
        # This is the max_step. Task is not completed (item not picked, not at depot).
        # Expected reward: -0.1 (move) - 20.0 (penalty for not completing) = -20.1
        # Done: True. Current_step: 2
        _, reward2, done2, _ = self.env.step(1) # Move Down again

        self.assertAlmostEqual(reward2, -0.1 - 20.0, delta=0.001,
                               msg="Reward should include movement cost and max_steps penalty")
        self.assertTrue(done2, "Episode should be done due to max_steps")
        self.assertEqual(self.env.current_step, 2) # current_step will be max_steps_per_episode
        self.assertFalse(self.env._get_state()['all_items_collected_phase'],
                         "Item not picked, so not in return phase")

        # Ensure task was not actually completed by checking conditions for completion penalty
        # Completion requires: all items picked AND robot at depot. If not, penalty applies.
        all_items_picked = all(self.env._get_state()['items_picked'])
        at_depot = np.array_equal(self.env._get_state()['robot_pos'], self.env.depot_pos)
        task_completed_conditions_met = all_items_picked and at_depot
        self.assertFalse(task_completed_conditions_met,
                         "Task should not be flagged as complete if max steps hit without completion")

        # Restore original max_steps if other tests might use self.env without a full reset
        # self.env.max_steps_per_episode = original_max_steps
        # However, setUp for each test re-initializes self.env, so this restoration isn't strictly needed here.

    # Test methods will be added below this line in subsequent steps.

if __name__ == '__main__':
    # This allows running the test file directly using 'python Chapter3/test_robot_warehouse_env.py'
    # from the project root directory.
    unittest.main()
