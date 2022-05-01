class fermentation_env2:
    """ fermentation simulation environment
    """

    def __init__(self, obj, initial_state=None):
        if initial_state is None:
            initial_state = [0.05, 0, 0, 30, 5, 0.7]
        self.done = False
        self.obj = obj
        self.initial_state = initial_state
        self.obj.update_coefficients(self.initial_state[0],
                                     self.initial_state[3],
                                     self.initial_state[4],
                                     98,
                                     self.initial_state[1],
                                     self.initial_state[2],
                                     0,
                                     self.initial_state[5])
        self.state = initial_state + [0] + [0, 0, self.obj.glucose_consumption(self.initial_state[0],
                                                                               self.initial_state[3], 0,
                                                                               self.initial_state[5])]
        self.glucose_consumed = 0
        self.glucose_added = 0  # self.initial_state[3]
        self.b = -0.13363 * 1000   # cost per g/L
        self.c = 1.28739   # product profit per g/L
        self.num_observation = len(self.state)
        self.num_action = 1

    def reset(self):
        self.done = False
        # self.state = self.initial_state + [0]
        self.obj.update_coefficients(self.initial_state[0],
                                     self.initial_state[3],
                                     self.initial_state[4],
                                     98,
                                     self.initial_state[1],
                                     self.initial_state[2],
                                     0,
                                     self.initial_state[5])
        self.state = self.initial_state + [0] + [0, 0, self.obj.glucose_consumption(self.initial_state[0],
                                                                                    self.initial_state[3], 0,
                                                                                    self.initial_state[5])]
        self.glucose_consumed = 0
        self.glucose_added = 0  # self.initial_state[3]
        return self.state

    def compute_reward(self, cumulative_feed, state):
        titer = state[1]
        total_CA = state[1] * state[-1]
        oil_consumed = cumulative_feed * 917 + self.initial_state[3] * state[-1] - state[3] * state[-1]
        yield_CA = total_CA / oil_consumed
        pv = titer / self.obj.harvest_time
        man_cost = (2 / pv + 1 / yield_CA) / 1000  # $/(g h)
        total_reward = - man_cost * self.obj.harvest_time  # * state[-1] * state[1]
        return total_reward

    def step(self, action):

        next_state = self.obj.predict(self.state[0], self.state[1], self.state[2], self.state[3], self.state[4],
                                      self.state[5], action[0], self.state[-4] * 4) + [self.state[-4] + 1]
        self.glucose_added += action[0] * self.obj.delta_t
        self.done = True if self.obj.harvest_time / self.obj.delta_t + 1 == next_state[-1] else False
        # reward = self.b * action[0] + self.c * (next_state[1] - self.state[1])
        reward = -(self.state[3] - 20) ** 2
        _next_state = next_state + [self.obj.dX_f, self.obj.dC, self.obj.dS]
        self.state = _next_state

        return _next_state, reward, self.done, None


class mechanism:
    """ mechanism model as dynamics for RL
    """

    def __init__(self, harvest_time=140):

        self.alpha_L = 0.127275
        self.c_max = 130.901733
        self.K_iN = 0.12294103
        self.K_iS = 612.178130
        self.K_iX = 59.9737695
        self.K_N = 0.02000201
        self.K_O = 0.33085322
        self.K_S = 0.0430429
        self.K_SL = 0.02165744
        self.m_s = 0.02252332
        self.r_L = 0.47917813
        self.V_evap = 2.6 * 1e-03
        self.Y_cs = 0.682572
        self.Y_ls = 0.3574429
        self.Y_xn = 10.0
        self.Y_xs = 0.2385559
        self.beta_LC_max = 0.14255192
        self.mu_max = 0.3844627
        # set up the kinetic constants
        self.dt = 0.1  # control in every hour
        # control variable
        # F_B
        # glucose concentration in glucose feed
        self.S_F = 917
        self.harvest_time = harvest_time
        self.delta_t = 4
        self.real_measurement = [0.0, 5.0, 10.0, 23.0, 28.0, 34.0, 47.5, 55.0, 72.0, 80.0, 95.0, 102.0, 120.0, 140.0]

    def update_coefficients(self, X_f, S, N, O, C, L, F_S, V):

        self.beta_LC = self.K_iN / (self.K_iN + N) * S / (S + self.K_S) * self.K_iS / (self.K_iS + S) * O / (
                self.K_O + O) * self.K_iX / (self.K_iX + X_f) * (1 - C / self.c_max) * self.beta_LC_max
        self.q_C = 2 * (1 - self.r_L) * self.beta_LC
        self.beta_L = self.r_L * self.beta_LC - self.K_SL * L / (L + X_f) * O / (O + self.K_O)
        self.mu = self.mu_max * S / (S + self.K_S) * self.K_iS / (self.K_iS + S) * N / (self.K_N + N) * O / (
                self.K_O + O) / (1 + X_f / self.K_iX)
        self.q_S = 1 / self.Y_xs * self.mu + O / (O + self.K_O) * S / (
                self.K_S + S) * self.m_s + 1 / self.Y_cs * self.q_C + 1 / self.Y_ls * self.beta_L
        self.F_B = V / 1000 * (7.14 / self.Y_xn * self.mu * X_f + 1.59 * self.q_C * X_f)
        self.D = (self.F_B + F_S) / V

    def lipid_free_cell_growth(self, X_f, V):
        self.dX_f = self.dt * (self.mu * X_f - (self.D - self.V_evap / V) * X_f)
        return self.dX_f + X_f

    def citrate_accumulation(self, X_f, C, V):
        self.dC = self.dt * (self.q_C * X_f - (self.D - self.V_evap / V) * C)
        return self.dC + C

    def lipid_accumulation(self, X_f, L, V):
        self.dL = self.dt * ((self.alpha_L * self.mu + self.beta_L) * X_f - (self.D - self.V_evap / V) * L)
        return self.dL + L

    def glucose_consumption(self, X_f, S, F_S, V):
        self.dS = - self.dt * (self.q_S * X_f - F_S / V * self.S_F + (self.D - self.V_evap / V) * S)
        return self.dS + S

    def nitrogen_consumption(self, X_f, N, V):
        self.dN = - self.dt * (1 / self.Y_xn * self.mu * X_f + (self.D - self.V_evap / V) * N)
        return max(self.dN + N, 0)

    def volume_change(self, F_S, V):
        self.dV = self.dt * (self.F_B + F_S - self.V_evap)
        return self.dV + V

    def one_step_predict(self, X_f, C, L, S, N, V, F_S, O):
        next_X_f = X_f
        next_C = C
        next_L = L
        next_S = S
        next_N = N
        next_V = V
        for i in range(int(self.delta_t / self.dt)):
            self.update_coefficients(next_X_f, next_S, next_N, O, next_C, next_L, F_S, next_V)
            next_X_f = max(self.lipid_free_cell_growth(next_X_f, next_V), 0)
            next_C = max(self.citrate_accumulation(next_X_f, next_C, next_V), 0)
            next_L = max(self.lipid_accumulation(next_X_f, next_L, next_V), 0)
            next_S = max(self.glucose_consumption(next_X_f, next_S, F_S, next_V), 0)
            next_N = max(self.nitrogen_consumption(next_X_f, next_N, next_V), 0)
            next_V = max(self.volume_change(F_S, next_V), 0)
        return [next_X_f, next_C, next_L, next_S, next_N, next_V]

    def predict(self, X_f, C, L, S, N, V, F_S, t):
        O = self.dissolve_oxygen(t)
        out = self.one_step_predict(X_f, C, L, S, N, V, F_S, O)
        return out

    def dissolve_oxygen(self, t):
        oxygen_data = [98.58750, 71.34750, 43.57285, 28.51669, 24.61062, 24.58139, 28.96252, 30.63478, 24.34490,
                       29.33842, 31.89646, 34.51994, 36.03211, 36.16354]

        index = 0
        time_w = -1
        for i, time in enumerate(self.real_measurement):
            if time - t > 0:
                index = i - 1
                time_w = time
                break
        return oxygen_data[index]
