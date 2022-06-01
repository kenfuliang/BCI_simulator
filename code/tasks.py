import json
import numpy as np
class Task(object):
    def __init__(self):
        #print("####Start task Construction####")
        self.time_duration = 3000 #(ms)
        self.task_states = {
                'begin'     : 1,
                'active'    : 2,
                'hold'      : 3,
                'success'   : 4,
                'fail'      : 5,
                'end'       : 6 }

        self.black = [0, 0, 0]
        self.green = [0, 255, 0]
        self.red   = [255, 0, 0]
        self.blue  = [0, 0, 255]
        self.white = [255, 255, 255]
        self.yellow = [100, 100, 0]

        self.size_cursor = int(20)
        self.cursor_color = self.white
        self.target_color = self.green
        self.target_type = 'square'

        self.time_success = 5
        self.time_fail = 1000
        self.time_begin = 5
        self.time_end = 10

        self.cursor_vel_scale = 1

        self.reset()

    def summary(self):
        raise NotImplementedError


    def reset(self):
        self.trial_count = int(1)
        self.task_state = 1
        self.cursor_on_target = False
        self.counter_trial = int(1)
        self.counter_hold = 0
        self.counter_begin = 0
        self.counter_success = 0
        self.counter_fail = 0
        self.counter_end = 0
        self.counter_duration = 0
        self.pos_cursor = np.array([0,0])
        self.vel_cursor = np.array([0,0]) 
        self.pos_target = np.array([0,0])

    
    def is_cursor_on_target(self,cursor,target,window):
        if self.target_type=='square':
            return target[0]-window/2<cursor[0]<target[0]+window/2 and target[1]-window/2<cursor[1]<target[1]+window/2
        elif self.target_type=='circle':
            return np.linalg.norm(cursor-target) < window

    def gen_new_target(self,back2Center) :
        raise NotImplementedError

    def update_if_fail(self):
        raise NotImplementedError

    def update(self,pos_cursor_i,PV_fail):
        dt         =      self.dt
        black      =      self.black
        green      =      self.green
        red        =      self.red
        blue       =      self.blue
        white      =      self.white
        yellow     =      self.yellow

        task_states = self.task_states
        task_state = self.task_state

        counter_trial      =     self.counter_trial
        counter_hold       =     self.counter_hold
        counter_begin      =     self.counter_begin
        counter_success    =     self.counter_success
        counter_fail       =     self.counter_fail
        counter_end        =     self.counter_end
        counter_duration   =     self.counter_duration
        time_success       =     self.time_success
        time_fail          =     self.time_fail
        time_begin         =     self.time_begin
        time_end           =     self.time_end
        time_duration      =     self.time_duration
        time_hold          =     self.time_hold

        acceptance_window = self.acceptance_window

        pos_target_i    = self.pos_target
        target_color_i  = self.target_color
        cursor_color_i  = self.cursor_color


        # change the cursor color if PV is not working
        if PV_fail:
            cursor_color_i=red
        else:
            cursor_color_i=white

        # update task state
        if task_state == task_states['begin'] :

            counter_begin += dt

            if counter_begin >= time_begin :
                task_state = task_states['active']
                counter_begin = 0
                pos_target_i = self.gen_new_target()
                target_color_i = green

        elif task_state == task_states['active'] :
            counter_duration += dt
            cursor_on_target = self.is_cursor_on_target(pos_cursor_i, pos_target_i, acceptance_window)

            if cursor_on_target :
                task_state = task_states['hold']
                counter_hold += dt
                target_color_i = yellow

            else:
                if counter_duration >= time_duration :
                    counter_duration = 0
                    task_state = task_states['fail']
                    target_color_i = red

        elif task_state == task_states['hold'] :
            counter_duration += dt
            cursor_on_target = self.is_cursor_on_target(pos_cursor_i, pos_target_i, acceptance_window)

            if not cursor_on_target :
                counter_hold = 0
                if counter_duration >= time_duration:
                    task_state = task_states['fail']
                    target_color_i = red
                else:
                    task_state = task_states['active']
                    target_color_i = green

            else:
                counter_hold += dt

                if counter_hold >= time_hold :
                    task_state = task_states['success']
                    counter_hold = 0

        elif task_state == task_states['success'] :

            counter_success += dt

            if counter_success >= time_success :

              task_state = task_states['end']
              counter_end += dt

        elif task_state == task_states['fail'] :

            counter_fail += dt

            if counter_fail >= time_fail :
                self.update_if_fail()

                task_state = task_states['end']
                counter_fail = 0

        elif task_state == task_states['end'] :

            counter_hold = 0
            counter_begin = 0
            counter_success = 0
            counter_fail = 0
            counter_duration = 0

            counter_end += dt

            if counter_end >= time_end :
                task_state = task_states['begin']
                counter_trial += 1
                counter_end = 0

        
        #print(counter_trial,counter_duration,task_state)
        self.counter_trial     = counter_trial
        self.counter_hold      = counter_hold
        self.counter_begin     = counter_begin
        self.counter_success   = counter_success
        self.counter_fail      = counter_fail
        self.counter_end       = counter_end
        self.counter_duration  = counter_duration

        # write output signals
        self.vel_cursor = pos_cursor_i - self.pos_cursor 
        self.pos_cursor = pos_cursor_i
        self.pos_target = pos_target_i
        self.cursor_color = cursor_color_i
        self.target_color = target_color_i
        self.task_state = task_state
        #self.tick_count = pNumTicks[0]
        self.trial_count = counter_trial


    def force(self,pos_cursor,pos_target,cursor_color,target_color):
        self.pos_cursor = pos_cursor
        self.pos_target = pos_target
        self.cursor_color = cursor_color
        self.target_color = target_color




class CenterOutTask(Task):
    def __init__(self,target_list=[]
                 ,dt=1
                 ,radius=80
                 ,time_hold=500
                 ,acceptance_window=40):
        super().__init__()

        #print("####Start centerOut_task Construction####")
        self.target_list = target_list
        self.radius = radius
        self.time_hold = time_hold #(ms)
        self.acceptance_window = acceptance_window
        #self.size_target = int(self.acceptance_window)
        self.dt = dt
        self.back2Center = False

    def summary(self):
        print("#### Summary of task ####")
        print("     dt",self.dt)
        print("     target_radius", self.radius)
        print("     acceptance_window", self.acceptance_window)
        print("     time_hold", self.time_hold)

    def gen_new_target(self):
        if len(self.target_list)!=0:
            return self.target_list.pop(0)
        else:
            return self.gen_random_target()
        

    def gen_random_target(self):
        back2Center = self.back2Center

        radius = self.radius
        if radius==120:
            target = np.array([[120,0],[84.85,84.85],[0,120],[-84.85,84.85],[-120,0],[-84.85,-84.85],[0,-120],[84.85,-84.85]])# (mm)
        elif radius==80:
            target = np.array([[80,0],[56.57,56.57],[0,80],[-56.57,56.57],[-80,0],[-56.57,-56.57],[0,-80],[56.57,-56.57]]) # (mm) in Gilja 2012
        rt_pos = np.array([0,0]) if back2Center else target[np.random.randint(len(target))]

        #self.back2Center = np.random.randint(2)#not back2Center
        self.back2Center = not back2Center

        return rt_pos

    def update_if_fail(self):
        self.back2Center = True


class PinballTask(Task):
    def __init__(self
                ,acceptance_window=40
                ,dt=1):
        #print("####Start pinball task Construction####")
        super().__init__()

        self.time_hold = 750 #(ms)
        self.workspace_size= 200 # 200x200 (mm)^2
        self.minimum_distance = 40
        self.maximum_distance = 140
        self.dt = dt
        self.acceptance_window = acceptance_window
        #self.size_target = int(self.acceptance_window)

    def summary(self):
        print("#### Summary of task ####")
        print("     dt",self.dt)
        print("     workspace_size", self.workspace_size)
        print("     acceptance_window", self.acceptance_window)
        print("     time_hold", self.time_hold)


    def gen_new_target(self):
        valid = False
        pos_target = self.pos_target
        workspace_size = self.workspace_size
        minimum_distance = self.minimum_distance
        maximum_distance = self.maximum_distance

        while (not valid):
            new_pos_target = np.random.uniform(0,workspace_size,2) - self.workspace_size/2
            distance = np.linalg.norm(pos_target - new_pos_target)
            if (distance > minimum_distance) and (distance < maximum_distance):
                valid = True

        return new_pos_target

    def update_if_fail(self):
        return



class DelayedCenterOutTask(CenterOutTask):
    def __init__(self
                ,time_delay=100
                ,**kwargs
                ):
        super().__init__(**kwargs)
        self.go_cue = False
        self.time_delay = time_delay
        self.last_target = None
        self.task_states = {
                'begin'     : 1,
                'delay'     : 2,
                'active'    : 3,
                'hold'      : 4,
                'success'   : 5,
                'fail'      : 6,
                'end'       : 7
                }
        self.reset()

    def reset(self):
        self.counter_delay = 0
        self.go_cue = False
        self.last_target = np.array([0,0])
        super().reset()

    def update(self,pos_cursor_i,PV_fail):
        dt         =      self.dt
        black      =      self.black
        green      =      self.green
        red        =      self.red
        blue       =      self.blue
        white      =      self.white
        yellow     =      self.yellow

        task_states = self.task_states
        task_state = self.task_state

        counter_trial      =     self.counter_trial
        counter_hold       =     self.counter_hold
        counter_begin      =     self.counter_begin
        counter_delay      =     self.counter_delay
        counter_success    =     self.counter_success
        counter_fail       =     self.counter_fail
        counter_end        =     self.counter_end
        counter_duration   =     self.counter_duration
        time_success       =     self.time_success
        time_fail          =     self.time_fail
        time_begin         =     self.time_begin
        time_end           =     self.time_end
        time_duration      =     self.time_duration
        time_hold          =     self.time_hold
        time_delay         =     self.time_delay

        acceptance_window = self.acceptance_window

        pos_target_i    = self.pos_target
        target_color_i  = self.target_color
        cursor_color_i  = self.cursor_color
        go_cue          = self.go_cue
        last_target     = self.last_target


        # change the cursor color if PV is not working
        if PV_fail:
            cursor_color_i=red
        else:
            cursor_color_i=white

        # update task state
        if task_state == task_states['begin'] :

            counter_begin += dt

            if counter_begin >= time_begin :
                task_state = task_states['delay']
                counter_begin = 0
                pos_target_i = self.gen_new_target()
                target_color_i = green
                
        elif task_state == task_states["delay"]:
            counter_delay += dt
            cursor_in_wait_area = self.is_cursor_in_wait_area(pos_cursor_i, self.last_target, acceptance_window)

            if counter_delay >= time_delay :
                task_state = task_states['active']
                counter_delay = 0
                go_cue = True

            elif not cursor_in_wait_area:
                task_state = task_states["fail"]
                target_color_i = red
                counter_delay = 0
                
        elif task_state == task_states['active'] :
            counter_duration += dt
            cursor_on_target = self.is_cursor_on_target(pos_cursor_i, pos_target_i, acceptance_window)
            
            if cursor_on_target :
                task_state = task_states['hold']
                counter_hold += dt
                target_color_i = yellow

            else:
                if counter_duration >= time_duration :
                    counter_duration = 0
                    task_state = task_states['fail']
                    target_color_i = red

        elif task_state == task_states['hold'] :
            counter_duration += dt
            cursor_on_target = self.is_cursor_on_target(pos_cursor_i, pos_target_i, acceptance_window)

            if not cursor_on_target :
                counter_hold = 0
                if counter_duration >= time_duration:
                    task_state = task_states['fail']
                    target_color_i = red
                else:
                    task_state = task_states['active']
                    target_color_i = green

            else:
                counter_hold += dt

                if counter_hold >= time_hold :
                    task_state = task_states['success']
                    counter_hold = 0

        elif task_state == task_states['success'] :

            counter_success += dt

            if counter_success >= time_success :

              task_state = task_states['end']
              counter_end += dt

        elif task_state == task_states['fail'] :

            counter_fail += dt

            if counter_fail >= time_fail :
                self.update_if_fail()

                task_state = task_states['end']
                counter_fail = 0

        elif task_state == task_states['end'] :

            counter_hold = 0
            counter_begin = 0
            counter_delay = 0
            counter_success = 0
            counter_fail = 0
            counter_duration = 0

            counter_end += dt

            last_target = pos_target_i

            if counter_end >= time_end :
                task_state = task_states['begin']
                counter_trial += 1
                counter_end = 0
                go_cue = False

        
        #print(counter_trial,counter_duration,task_state)
        self.counter_trial     = counter_trial
        self.counter_hold      = counter_hold
        self.counter_begin     = counter_begin
        self.counter_delay     = counter_delay
        self.counter_success   = counter_success
        self.counter_fail      = counter_fail
        self.counter_end       = counter_end
        self.counter_duration  = counter_duration
        self.go_cue            = go_cue
        self.last_target       = last_target

        # write output signals
        self.vel_cursor = pos_cursor_i - self.pos_cursor 
        self.pos_cursor = pos_cursor_i
        self.pos_target = pos_target_i
        self.cursor_color = cursor_color_i
        self.target_color = target_color_i
        self.task_state = task_state
        #self.tick_count = pNumTicks[0]
        self.trial_count = counter_trial

    def is_cursor_in_wait_area(self, pos_cursor_i, pos_target_i, acceptance_window=10):
        return self.is_cursor_on_target(pos_cursor_i, pos_target_i, acceptance_window)

