import os
from overrides import overrides
from action.base import BaseAction
from irm_task.explore_multiple_targets import ExploreMultipleTargetsTask
from irm_task.base import ExecutionMode
from prompts import get_default_guidance
from logging_utils import get_logger
logger = get_logger(__name__)

class GotoTargetPointAction(BaseAction):
    def __init__(self, drm, irm, args, binder_config):
        super().__init__(drm, irm, args, binder_config)
        self.should_interrupt = False
        self.interrupt_reason = None

    @overrides
    def predict_irm_tasks(self):
        seen_objects = list(self.drm.memory["object_positions"].keys())
        unseen_objects = list(set(self.drm.memory["target_objects"]) - set(seen_objects))
        current_target_object = self.args['target_object']

        if len(seen_objects) == 0 and len(unseen_objects) == 0:
            return None
        
        # Guidance prompt G from the DRM (paper Eq. 1); fall back to the default phase
        # guidance when the DRM does not supply one.
        guidance = self.args.get("guidance") or get_default_guidance(
            "goto_target_point", self.drm.memory["target_objects"]
        )
        irm_config = {
            'action': 'goto_target_point',
            'seen_objects': seen_objects,
            'unseen_objects': unseen_objects,
            'current_target_object': current_target_object,
            'guidance': guidance,
            'verbose': True,
            'activate_only_on_moving_base': True,
            'log_dir': f"{self.agent.log}/{self.drm.drm_call_count}_goto_target_point",
        }
        localize_objects_task = ExploreMultipleTargetsTask(
            irm=self.irm,
            robot=self.irm.robot,
            agent=self.drm.agent,
            get_drm_memory_fn=self.drm.get_drm_memory,
            vlm1_socket=self.irm.vlm1_socket,
            image_buffer=self.irm._image_buffer,
            _ee_image_buffer=self.irm._ee_image_buffer,
            fps=self.irm._fps,
            config=irm_config
        )

        return [localize_objects_task]
    

    def handle_irm_event(self, event):
        # Act on the IRM execution mode m_t (paper Eq. 3), not the event name.
        mode = event.get("mode")
        data = event.get("data", {})
        if mode == ExecutionMode.REPLAN:
            # task-critical change (current target appeared/moved) -> stop, DRM replans
            self.should_interrupt = True
            self.interrupt_reason = data.get("reason", "IRM REPLAN")
        elif mode == ExecutionMode.CONTINUE and "target_point" in data:
            # non-critical object update -> refresh memory R_t, keep going
            self.drm.memory["object_positions"][data["target_object"]] = {
                "xyz": data["target_point"],
                "image_id": data["image_idx"],
                "score": data["score"],
            }

    @overrides
    def run(self):
        if not self.args['target_object'] in self.drm.memory['object_positions']:
            self.drm.action_history.append({
            "action": "goto_target_point",
            "arguments": {
                "target_object": self.args['target_object'],
            },
            "status": "failed - cannot go to target point before localizing. You should first execute 'exploration'.",
            "result": None,
            })
            return
        
        return super().run()


            
    @overrides
    def before_action_task(self):
        if self.enable_irm:
            irm_tasks = self.predict_irm_tasks()
            if irm_tasks is not None:
                self.irm.set_tasks(irm_tasks)
        self.target_object = self.args['target_object']
        self.target_point = self.drm.memory['object_positions'][self.target_object]['xyz']

    def should_interrupt_navigation(self):
        return self.should_interrupt
    
    @overrides
    def execute(self):
        log_dir = f"{self.agent.log}/{self.drm.drm_call_count}_goto_target_point"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        finished, object_moved_or_removed = self.move_to_reachable_nearest_point(
            self.target_point, 
            visualize=True, 
            interrupt_condition=self.should_interrupt_navigation, 
            log_dir=log_dir,
            target_object=self.target_object
        )
        return finished, object_moved_or_removed
    
    @overrides
    def update_drm_history(self, result=None):
        if result is None:
            finished, object_moved_or_removed = False, False
        else:
            finished, object_moved_or_removed = result
        
        if self.should_interrupt:
            self.drm.action_history.append({
                "action": "goto_target_point",
                "arguments": {
                    "target_object": self.args['target_object'],
                },
                "status": "failed - Target object moved from the original position. You should execute 'goto_target_point' again.",
                "result": None,
            })
        elif object_moved_or_removed:
            self.drm.action_history.append({
                "action": "goto_target_point",
                "arguments": {
                    "target_object": self.args['target_object'],
                },
                "status": "failed - Target object moved or removed during navigation. You should execute 'goto_target_point' or 'explore_environment' again.",
                "result": None,
            })
        else:
            self.drm.action_history.append({
                "action": "goto_target_point",
                "arguments": {
                    "target_object": self.args['target_object'],
                },
                "status": "success",
                "result": None,
            })
