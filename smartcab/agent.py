import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qValues = {}
        self.gamma = 0.30 #(discount rate gamma)
        #self.alpha = 1.0 #(learning rate)
        #self.epsilon = 0.50 #(exploration probability)
        self.t = 1
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.n_run = 1

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.n_run += 1

    def getQValue(self, state, action):
        qvalue = self.qValues.get((state,action))
        if qvalue != None:
            return qvalue
        self.qValues[(state,action)] = 10.0
        return self.qValues.get((state,action))

    def getBestValueFromQValues(self, state):
        best_qvalue = float('-inf')
        for action in self.env.valid_actions:
            current_qvalue =  self.getQValue(state, action)
            if current_qvalue > best_qvalue:
                best_qvalue = current_qvalue
        return best_qvalue

    def getBestActionFromQValues(self, state):
        best_action = None
        best_qvalue = self.getBestValueFromQValues(state)
        best_actions_list = list()
        for action in self.env.valid_actions:
            current_qvalue =  self.getQValue(state, action)
            if current_qvalue == best_qvalue:
                best_actions_list.append(action)
        if (len(best_actions_list)>0):
            best_action = random.choice(best_actions_list)
        return best_action

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # this variable will code if the next_waypoint suggested by route planner is a ok decision to take or not
        # based on the traffic rules suggested in the probelm
        # these are essentially based on three variables , state of traffic light, direction of oncoming tarffic if any
        # and direction of traffic coming from left.
        # there were two choices - one was to code the problem with all these variables in raw form which would ahve increased the
        # state space. So I decided to combine them into a variable "state_nextway_ok" which is Treu if taking planner
        # suggested route will meet traffic rules or not

        #self.state = (self.next_waypoint,inputs['light'], inputs['oncoming'], inputs['left'], state_nextway_ok)
        self.state = (self.next_waypoint, inputs['light'])

        # TODO: Select action according to your policy
        # Q1: Random for Q1
        #action = random.choice(self.env.valid_actions)

        # Q3: Always take most optimal action i.e. no exploration
        #action = self.getBestActionFromQValues(self.state)


        # final model for choosing action
        epsilon = 1.0/(1.0+self.n_run)
        #epsilon = 0.0  #done to check the state space exploration under Q3
        if random.random() < epsilon:
            action = random.choice(self.env.valid_actions)
        else:
            action = self.getBestActionFromQValues(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)


        # TODO: Learn policy based on state, action, reward
        alpha = 100.0/(100.0+self.t)
        self.t += 1
        if self.last_state != None:
            sample = self.last_reward + self.gamma *self.getBestValueFromQValues(self.state)
            pre_Qvalue = self.getQValue(self.last_state, self.last_action)
            new_Qvalue = (1-alpha) * pre_Qvalue + alpha * sample
            self.qValues[(self.last_state, self.last_action)] = new_Qvalue

        self.last_state = self.state
        self.last_reward = reward
        self.last_action = action

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #print "LearningAgent.update(): deadline = {}, state = {}, action = {}, reward = {}".format(deadline, self.state, action, reward)  # [debug]
        #print destination

class FinalAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, qValues):
        super(FinalAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.qValues = qValues
        #print "Learnt qValues: {}".format(self.qValues)


    def reset(self, destination=None):
        self.planner.route_to(destination)


    def getQValue(self, state, action):
        qvalue = self.qValues.get((state,action))
        if qvalue != None:
            return qvalue
        return 0.0

    def getBestValueFromQValues(self, state):
        best_qvalue = float('-inf')
        for action in self.env.valid_actions:
            current_qvalue =  self.getQValue(state, action)
            if current_qvalue > best_qvalue:
                best_qvalue = current_qvalue
        return best_qvalue

    def getBestActionFromQValues(self, state):
        best_action = None
        best_qvalue = self.getBestValueFromQValues(state)
        best_actions_list = list()
        for action in self.env.valid_actions:
            current_qvalue =  self.getQValue(state, action)
            if current_qvalue == best_qvalue:
                best_actions_list.append(action)
        if (len(best_actions_list)>0):
            best_action = random.choice(best_actions_list)
        return best_action

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        location = self.env.agent_states[self]['location']
        destination = self.env.agent_states[self]['destination']

        # TODO: Update state

        self.state = (self.next_waypoint, inputs['light'])

        # TODO: Select action according to your policy
        action = self.getBestActionFromQValues(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        #print "FinalAgent.update(): deadline = {}, state = {}, action = {}, reward = {}".format(deadline, self.state, action, reward)  # [debug]
        if reward < 0.0:
            print "TRAFFIC VIOLATION"

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.01)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    print "qValue: nos: {}  list: {}".format(len(a.qValues),a.qValues )

    #FInal test run
    print "Running the agent again with learnt Q policy. This phase has no learning"
    e = Environment()
    b = e.create_agent(FinalAgent, a.qValues)
    e.set_primary_agent(b, enforce_deadline=True)  # set agent to track
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
