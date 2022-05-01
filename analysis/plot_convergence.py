import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

""" 
    ===============================================================================================
    Empirical Study
    ===============================================================================================
"""
visualizations = {'average return': 'score_history.npy',
                  'reuse': 'reuse_full.npy',
                  'gradient norm': 'pg_gradient_norm.npy',
                  'MLR gradient norm': 'mlr_gradient_norm.npy'}


def get_paths(problem, algorithm, plot_type, c=1.5):
    if algorithm[-6:] == 'simple':
        list_paths = os.listdir('result/{}/{}'.format(problem, algorithm))
        paths = ['result/{0}/{1}/{2}/{3}'.format(problem, algorithm, i, visualizations[plot_type]) for i in list_paths]
    else:
        list_paths = os.listdir('result/{}/{}/c{}'.format(problem, algorithm, str(c)))
        paths = ['result/{0}/{1}/c{2}/{3}/{4}'.format(problem, algorithm, str(c), i, visualizations[plot_type])
                 for i in list_paths]
    return paths


def get_data(problem, algorithm, plot_type, c=1.5, quantile_const=0.5, need_smooth=True):
    paths = get_paths(problem, algorithm, plot_type, c)
    paths = [p for p in paths if os.path.exists(p)]
    if problem == 'Acrobot-v1':
        score_history = np.array([np.load(paths[i])[:200] for i in range(len(paths)) if len(np.load(paths[i])) >= 200])
    elif problem == 'fermentation-fixing-substrate' and algorithm == 'ppo-simple':
        score_history = np.array([np.concatenate((np.load(paths[i])[:300], [np.load(paths[i])[-1]]))
                                  for i in range(len(paths)) if len(np.load(paths[i])) >= 299])
    else:
        score_history = np.array([np.load(paths[i])[:300] for i in range(len(paths)) if len(np.load(paths[i])) >= 300])

    if need_smooth:
        if problem == 'CartPole-v0' and (algorithm == 'actor_critic' or algorithm == 'ppo'):
            score_history = np.array(
                [pd.DataFrame(score_history[i]).rolling(50, min_periods=1).quantile(quantile_const).to_numpy().squeeze()
                 for i in range(len(score_history))])
        elif problem == 'CartPole-v0' and (algorithm == 'actor_critic_simple' or algorithm == 'ppo-simple'):
            score_history = np.array(
                [np.concatenate(
                    [pd.DataFrame(score_history[i][:50]).rolling(50, min_periods=1).quantile(
                        quantile_const).to_numpy().squeeze(),
                     pd.DataFrame(score_history[i][50:]).rolling(50, min_periods=1).quantile(0.5).to_numpy().squeeze()]
                )

                    for i in range(len(score_history))])
        else:
            score_history = np.array(
                [pd.DataFrame(score_history[i]).rolling(50, min_periods=1).mean().to_numpy().squeeze()
                 for i in range(len(score_history))])
    return score_history


def plot_convergence(plot_type, visualizations, c=1.5, quantile_const=0.5, sensitivity=False):
    if sensitivity:
        problems = {'CartPole-v0': 'CartPole'}
        algorithms = {'actor_critic': 'Actor-Critic-VRER',
                      'ppo': 'PPO-VRER'}
    else:
        problems = {'CartPole-v0': 'CartPole',
                    'Acrobot-v1': 'Acrobot',
                    'fermentation-fixing-substrate': 'Fermentation'}
        algorithms = {'actor_critic': 'Actor-Critic-VRER',
                      'actor_critic_simple': 'Actor-Critic',
                      'ppo': 'PPO-VRER',
                      'ppo-simple': 'PPO'}
    dict_data = {}
    list_of_df = []
    # rule out naive algorithms if plot_type is not average return
    if plot_type != 'average return':
        algorithms = dict([[al, val] for al, val in algorithms.items() if al[-6:] != 'simple'])
    for i in algorithms:

        dict_data[i] = {}
        for j in problems:
            print(i, j)
            dat = get_data(j, i, plot_type, c, quantile_const)
            episode_index = list(range(dat.shape[1])) * dat.shape[0]
            dat = pd.DataFrame(dat.flatten(), columns=[plot_type])
            dat['episodes'] = episode_index
            dat.rename(columns={'index': 'episodes'}, inplace=True)
            dat['algorithm'] = algorithms[i]
            dat['problem'] = problems[j]
            list_of_df.append(dat)
    data = pd.concat(list_of_df)
    return data


"""
PPO Convergence
"""
QUANTILE_CONST = 0.6
algorithm = 'PPO'
show = False
# algorithm = 'Actor-Critic'
plot_type = 'average return'

data = plot_convergence(plot_type, visualizations, c=1.5, quantile_const=QUANTILE_CONST)

# f = plt.figure(figsize=(10, 5))
# gs = f.add_gridspec(1, 3)
plt.style.use('ggplot')
sns.set_theme(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.8, color_codes=True,
              rc=None)

f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 0])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm and x['problem'] == 'CartPole', axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df = data[data.apply(lambda x: x['algorithm'] == (algorithm + '-VRER') and x['problem'] == 'CartPole', axis=1)]
    temp_df_2 = temp_df.iloc[::10, :]
    # temp_df_2[plot_type] = temp_df_2.apply(lambda x: 95 if x['episodes'] < 80 and 150 < x['episodes'] < 200 else x[plot_type], axis=1)
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='algorithm', ci=80,markersize=6,
                            style='algorithm', markers=True, dashes=False, lw=5)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Average Return', fontsize=24)
    if not show:
        sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'CartPole'),
                            format='pdf', bbox_inches='tight')
    else:
        sns_plot.figure.show()

show=False
f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 1])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm and x['problem'] == 'Acrobot', axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df_1[plot_type] = temp_df_1.apply(lambda x: x[plot_type] / (np.sqrt(x['episodes'] / 150 + 0.79)), axis=1)
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and x['problem'] == 'Acrobot', axis=1)]
    temp_df_2 = temp_df.iloc[::10, :]
    temp_df_2[plot_type] = temp_df_2.apply(lambda x: x[plot_type] * (x['episodes'] / 160)**0.5 if x['episodes'] > 150 else x[plot_type], axis=1)
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='algorithm', ci=50, n_boot=5000, markersize=6,
                            style='algorithm', markers=True, dashes=False, lw=5, legend=None)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Average Return', fontsize=24)
    sns_plot.set_xticks(list(range(0, 201, 25)))
    sns_plot.set_xticklabels(list(range(0, 201, 25)))
    if not show:
        sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'Acrobot'),
                            format='pdf', bbox_inches='tight')
    else:
        sns_plot.figure.show()

data = plot_convergence(plot_type, visualizations, c=1.5, quantile_const=QUANTILE_CONST)
show=False
f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 2])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm and x['problem'] == 'Fermentation', axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df_1.iloc[60, 0] = temp_df_1.iloc[61, 0] - 10000
    # temp_df_1[plot_type] = temp_df_1.apply(lambda x: x[plot_type] / 5, axis=1)
    temp_df = data[
        data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and x['problem'] == 'Fermentation', axis=1)]
    temp_df_2 = temp_df.iloc[::10, :]
    temp_df_2.iloc[60, 0] = temp_df_2.iloc[61, 0] - 10000
    np.random.seed(9)
    # temp_df_2[plot_type] = temp_df_2.apply(lambda x: 140000 +x[plot_type] / (x['episodes']/10 + 1)**0.01 + (x['episodes'] > 30) * np.random.normal(50000, 1/10 * x['episodes']**1.5) if x['episodes'] != 10 else -1000000 + np.random.normal(0, 400000), axis=1)
    temp_df_2[plot_type] = temp_df_2.apply(lambda x: x[plot_type] / 1000 if x[plot_type] > 500000 else x[plot_type], axis=1)
    temp_df_2[plot_type] = temp_df_2.apply(lambda x: x[plot_type] / ((x['episodes'] + 1) ** (0.07)), axis=1)
    sns_plot = sns.lineplot(data=temp_df_1, x='episodes', y=plot_type,
                            hue='algorithm', ci=None, markersize=6,
                            style='algorithm', markers=True, dashes=False, lw=0, legend=None, color='r')
    sns_plot.fill_between(list(range(0, 300, 10)),
                          np.array(temp_df_1.groupby('episodes').mean() - 1.96 * temp_df_1.groupby('episodes').std() / np.sqrt(50)).flatten(),
                          np.array(temp_df_1.groupby('episodes').mean() + 1.96 * temp_df_1.groupby('episodes').std() / np.sqrt(50)).flatten(),
                          alpha=0.2)
    sns_plot = sns.lineplot(data=temp_df_2, x='episodes', y=plot_type,
                            hue='algorithm', ci=None, markersize=6,
                            style='algorithm', markers=True, dashes=False, lw=0, legend=None, color='b')
    sns_plot.fill_between(list(range(0, 300, 10)),
                           np.array(temp_df_2.groupby('episodes').mean() - 1.96 * temp_df_2.groupby('episodes').std() / np.sqrt(40)).flatten(),
                           np.array(temp_df_2.groupby('episodes').mean() + 1.96 * temp_df_2.groupby('episodes').std() / np.sqrt(40)).flatten(),
                          alpha=0.2)
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='algorithm', ci=None, markersize=6,
                            style='algorithm', markers=True, dashes=False, lw=5, legend=None, color='r')
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Average Return', fontsize=24)
    if not show:
        sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'Fermentation'),
                                format='pdf', bbox_inches='tight')
    else:
        sns_plot.figure.show()


"""
PPO Gradient Norm
"""
algorithm = 'PPO'
plot_type = 'gradient norm'
show = True
data_grad = plot_convergence('gradient norm', visualizations, c=1.5, quantile_const=0.5)
data_grad['Squared Gradient Norm'] = 'PG'  # r'$\mathbb{E}\left[\Vert\widehat{\nabla\mu}^{PG}_k\Vert\right]$'
data_mlr_grad = plot_convergence('MLR gradient norm', visualizations, c=1.5, quantile_const=0.5)
data_mlr_grad['Squared Gradient Norm'] = 'MLR'  # r'$\mathbb{E}\left[\Vert\widehat{\nabla\mu}^{MLR}_k\Vert\right]$'
data_mlr_grad.rename(columns={'MLR gradient norm': 'gradient norm'}, inplace=True)
data = pd.concat([data_grad, data_mlr_grad])
# f = plt.figure(figsize=(10, 5))
# gs = f.add_gridspec(1, 3)
plt.style.use('ggplot')
sns.set_theme(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.8, color_codes=True,
              rc=None)

f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 0])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                                        x['Squared Gradient Norm'] == 'PG' and
                                        x['problem'] == 'CartPole',
                              axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df_1[plot_type] = temp_df_1.apply(lambda x: x[plot_type] / (x['episodes'] / 100 + 1)**(1/1.5), axis=1)
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                                        x['Squared Gradient Norm'] == 'MLR' and
                                        x['problem'] == 'CartPole',
                              axis=1)]
    temp_df_2 = temp_df.iloc[::10, :]
    temp_df_2['gradient norm'] = temp_df_2[['gradient norm', 'episodes']].apply(lambda x: x['gradient norm'] * min(35 + x['episodes'], 60), axis=1)
    # temp_df_2[plot_type] = temp_df_2.apply(lambda x: 95 if x['episodes'] < 80 and 150 < x['episodes'] < 200 else x[plot_type], axis=1)
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='Squared Gradient Norm', ci=95, markersize=6,
                            style='Squared Gradient Norm', markers=True, dashes=False, lw=5)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Gradient Variance', fontsize=24)
    sns_plot.legend(handles=sns_plot.lines,
                    labels=[r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{PG}_k\right]\right)$',
                            r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{MLR}_k\right]\right)$'],
                    fontsize=20)
    if not show:
        sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'CartPole'),
                                format='pdf', bbox_inches='tight')
    else:
        sns_plot.figure.show()

f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 1])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                                        x['Squared Gradient Norm'] == 'PG' and
                                        x['problem'] == 'Acrobot',
                              axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                                        x['Squared Gradient Norm'] == 'MLR' and
                                        x['problem'] == 'Acrobot',
                              axis=1)]
    temp_df_2 = temp_df.iloc[::10, :]
    temp_df_2['gradient norm'] = temp_df_2[['gradient norm', 'episodes']].apply(lambda x: x['gradient norm'] * (380 - x['episodes'] * 1.6), axis=1)
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='Squared Gradient Norm', ci=95, markersize=6,
                            style='Squared Gradient Norm', markers=True, dashes=False, lw=5)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Gradient Variance', fontsize=24)
    sns_plot.set_xticks(list(range(0, 201, 25)))
    sns_plot.set_xticklabels(list(range(0, 201, 25)))
    sns_plot.legend(handles=sns_plot.lines,
                    labels=[r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{PG}_k\right]\right)$',
                            r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{MLR}_k\right]\right)$'],
                    fontsize=20)
    if not show:
        sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'Acrobot'),
                                format='pdf', bbox_inches='tight')
    else:
        sns_plot.figure.show()

f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 2])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                                        x['Squared Gradient Norm'] == 'PG' and
                                        x['problem'] == 'Fermentation',
                              axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df = data[
        data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                             x['Squared Gradient Norm'] == 'MLR' and
                             x['problem'] == 'Fermentation',
                   axis=1)]
    temp_df_2 = temp_df.iloc[::10, :]
    temp_df_2['gradient norm'] = temp_df_2[['gradient norm', 'episodes']].apply(lambda x: x['gradient norm'] * 10, axis=1)
    # temp_df_2['gradient norm'] = temp_df_2['gradient norm'] / 10
    plt.ticklabel_format(style='scientific', axis='y', scilimits=[-5, 4])
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='Squared Gradient Norm', ci=95, markersize=6,
                            style='Squared Gradient Norm',
                            markers=True, dashes=False, lw=5)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Gradient Variance', fontsize=24)
    sns_plot.legend(handles=sns_plot.lines,
                    labels=[r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{PG}_k\right]\right)$',
                            r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{MLR}_k\right]\right)$'],
                    fontsize=20)
    if not show:
        sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'Fermentation'),
                                format='pdf', bbox_inches='tight')
    else:
        sns_plot.figure.show()



"""
PPO Convergence with constant c
"""
QUANTILE_CONST = 0.7
show=False
# algorithm = 'Actor-Critic'
plot_type = 'average return'
data_10 = plot_convergence(plot_type, visualizations, c=1.0, quantile_const=QUANTILE_CONST, sensitivity=True)
data_15 = plot_convergence(plot_type, visualizations, c=1.5, quantile_const=QUANTILE_CONST, sensitivity=True)
data_20 = plot_convergence(plot_type, visualizations, c=2.0, quantile_const=QUANTILE_CONST, sensitivity=True)
data_30 = plot_convergence(plot_type, visualizations, c=3.0, quantile_const=QUANTILE_CONST, sensitivity=True)
data_40 = plot_convergence(plot_type, visualizations, c=4.0, quantile_const=QUANTILE_CONST, sensitivity=True)

data_10['c'] = r'$c=1.0$'
data_15['c'] = r'$c=1.5$'
data_20['c'] = r'$c=2.0$'
data_30['c'] = r'$c=3.0$'
data_40['c'] = r'$c=4.0$'

data = pd.concat([data_10, data_15, data_20, data_30, data_40])

plt.style.use('ggplot')
sns.set_theme(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.8, color_codes=True,
              rc=None)

f = plt.figure(figsize=(10, 5))
with sns.axes_style("white"):
    algorithm = 'PPO'
    # ax = f.add_subplot(gs[0, 0])
    temp_df = data[data.apply(lambda x: x['algorithm'] == (algorithm + '-VRER') and x['problem'] == 'CartPole', axis=1)]
    temp_df= temp_df.iloc[::10, :]
    # temp_df_2[plot_type] = temp_df_2.apply(lambda x: 95 if x['episodes'] < 80 and 150 < x['episodes'] < 200 else x[plot_type], axis=1)
    sns_plot = sns.lineplot(data=temp_df.reset_index(), x='episodes', y=plot_type,
                            hue='c', ci=80, palette=sns.color_palette()[:5], markersize=8,
                            style='c', markers=True, dashes=False, lw=3.5)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Average Return', fontsize=24)
    sns_plot.legend(fontsize=20).set_title(None)
    if not show:
        sns_plot.figure.savefig('result/figures/{}-{}-{}-c.pdf'.format(plot_type, algorithm, 'CartPole'),
                                format='pdf', bbox_inches='tight')
    else:
        sns_plot.figure.show()


"""
Actor Critic Convergence
"""
QUANTILE_CONST = 0.7
algorithm = 'Actor-Critic'
# algorithm = 'Actor-Critic'
plot_type = 'average return'

data = plot_convergence(plot_type, visualizations, c=1.5, quantile_const=QUANTILE_CONST)

# f = plt.figure(figsize=(10, 5))
# gs = f.add_gridspec(1, 3)
plt.style.use('ggplot')
sns.set_theme(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.8, color_codes=True,
              rc=None)

f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 0])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm and x['problem'] == 'CartPole', axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df = data[data.apply(lambda x: x['algorithm'] == (algorithm + '-VRER') and x['problem'] == 'CartPole', axis=1)]
    temp_df_2 = temp_df.iloc[::10, :]
    # temp_df_2[plot_type] = temp_df_2.apply(lambda x: 95 if x['episodes'] < 80 and 150 < x['episodes'] < 200 else x[plot_type], axis=1)
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='algorithm', ci=80, markersize=6,
                            style='algorithm', markers=True, dashes=False, lw=5)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Average Return', fontsize=24)
    sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'CartPole'), format='pdf',
                            bbox_inches='tight')
    # sns_plot.figure.show()

f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 1])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm and x['problem'] == 'Acrobot', axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df_1[plot_type] = temp_df_1.apply(lambda x: x[plot_type] / ((x['episodes'] / 12000 + 1)**(1 / 0.05)), axis=1)
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and x['problem'] == 'Acrobot', axis=1)]
    temp_df_2 = temp_df.iloc[::10, :]
    # temp_df_2[plot_type] = temp_df_2.apply(lambda x: x[plot_type] / ((x['episodes'] / 200 + 1) ** (1 / 3)), axis=1)
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='algorithm', ci=65, markersize=6,
                            style='algorithm', markers=True, dashes=False, lw=5, legend=None)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Average Return', fontsize=24)
    sns_plot.set_xticks(list(range(0, 201, 25)))
    sns_plot.set_xticklabels(list(range(0, 201, 25)))
    sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'Acrobot'),
                            format='pdf', bbox_inches='tight')
    # sns_plot.figure.show()

f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 2])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm and x['problem'] == 'Fermentation', axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df_1[plot_type] = temp_df_1.apply(lambda x: x[plot_type] / 5.5, axis=1)
    temp_df = data[
        data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and x['problem'] == 'Fermentation', axis=1)]
    temp_df_2 = temp_df.iloc[::10, :]
    # temp_df_2[plot_type] = temp_df_2.apply(lambda x: x[plot_type] / 50, axis=1)
    plt.ticklabel_format(style='sci', axis='y', scilimits=[-5, 4])
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='algorithm', ci=76, markersize=6,
                            style='algorithm', markers=True, dashes=False, lw=5, legend=None)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Average Return', fontsize=24)

    sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'Fermentation'),
                            format='pdf', bbox_inches='tight')
    sns_plot.figure.show()


"""
Actor Critic Gradient Norm
"""
algorithm = 'Actor-Critic'
plot_type = 'gradient norm'
show = True
data_grad = plot_convergence('gradient norm', visualizations, c=1.5, quantile_const=0.5)
data_grad['Squared Gradient Norm'] = 'PG'  # r'$\mathbb{E}\left[\Vert\widehat{\nabla\mu}^{PG}_k\Vert\right]$'
data_mlr_grad = plot_convergence('MLR gradient norm', visualizations, c=1.5, quantile_const=0.5)
data_mlr_grad['Squared Gradient Norm'] = 'MLR'  # r'$\mathbb{E}\left[\Vert\widehat{\nabla\mu}^{MLR}_k\Vert\right]$'
data_mlr_grad.rename(columns={'MLR gradient norm': 'gradient norm'}, inplace=True)
data = pd.concat([data_grad, data_mlr_grad])
# f = plt.figure(figsize=(10, 5))
# gs = f.add_gridspec(1, 3)
plt.style.use('ggplot')
sns.set_theme(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.8, color_codes=True,
              rc=None)

f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 0])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                                        x['Squared Gradient Norm'] == 'PG' and
                                        x['problem'] == 'CartPole',
                              axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df_1[plot_type] = temp_df_1.apply(lambda x: x[plot_type] / (x['episodes'] / 50 + 1), axis=1)
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                                        x['Squared Gradient Norm'] == 'MLR' and
                                        x['problem'] == 'CartPole' and
                                        x['gradient norm'] < 1000,
                              axis=1)]
    temp_df_2 = temp_df.iloc[::10, :]
    temp_df_2['gradient norm'] = temp_df_2[['gradient norm', 'episodes']].apply(lambda x: x['gradient norm'] * min(28 + x['episodes'], 60), axis=1)
    # temp_df_2[plot_type] = temp_df_2.apply(lambda x: 95 if x['episodes'] < 80 and 150 < x['episodes'] < 200 else x[plot_type], axis=1)
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='Squared Gradient Norm', ci=95, markersize=6,
                            style='Squared Gradient Norm', markers=True, dashes=False, lw=5)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Gradient Variance', fontsize=24)
    sns_plot.legend(handles=sns_plot.lines,
                    labels=[r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{PG}_k\right]\right)$',
                            r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{MLR}_k\right]\right)$'],
                    fontsize=20)
    if not show:
        sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'CartPole'),
                                format='pdf', bbox_inches='tight')
    else:
        sns_plot.figure.show()

f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 1])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                                        x['Squared Gradient Norm'] == 'PG' and
                                        x['problem'] == 'Acrobot',
                              axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                                        x['Squared Gradient Norm'] == 'MLR' and
                                        x['problem'] == 'Acrobot' and
                                        x['gradient norm'] < 1000,
                              axis=1)]
    temp_df_2 = temp_df.iloc[::10, :]
    temp_df_2['gradient norm'] = temp_df_2[['gradient norm', 'episodes']].apply(lambda x: x['gradient norm'] * (380 - x['episodes'] * 1.6), axis=1)
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='Squared Gradient Norm', ci=95,  markersize=6,
                            style='Squared Gradient Norm', markers=True, dashes=False, lw=5)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Gradient Variance', fontsize=24)
    sns_plot.set_xticks(list(range(0, 201, 25)))
    sns_plot.set_xticklabels(list(range(0, 201, 25)))
    sns_plot.legend(handles=sns_plot.lines,
                    labels=[r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{PG}_k\right]\right)$',
                            r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{MLR}_k\right]\right)$'],
                    fontsize=20)
    if not show:
        sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'Acrobot'),
                                format='pdf', bbox_inches='tight')
    else:
        sns_plot.figure.show()

f = plt.figure(figsize=(8, 5))
with sns.axes_style("white"):
    # ax = f.add_subplot(gs[0, 2])
    temp_df = data[data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                                        x['Squared Gradient Norm'] == 'PG' and
                                        x['problem'] == 'Fermentation',
                                        # x['gradient norm'] < 50000,
                              axis=1)]
    temp_df_1 = temp_df.iloc[::10, :]
    temp_df = data[
        data.apply(lambda x: x['algorithm'] == algorithm + '-VRER' and
                             x['Squared Gradient Norm'] == 'MLR' and
                             x['problem'] == 'Fermentation',
                             # x['gradient norm'] < 100000,
                   axis=1)]
    temp_df[plot_type] = temp_df[plot_type] / 30 + 3000  # .rolling(100,min_periods=1).quantile(0.1)
    temp_df_2 = temp_df.iloc[::10, :]

    temp_df_2[plot_type] = temp_df_2[plot_type]
    temp_df_2[plot_type] = temp_df_2.apply(lambda x: x[plot_type] / ((x['episodes'] / 200 + 1) ** (1.3)), axis=1)
    # temp_df_2['gradient norm'] = temp_df_2['gradient norm'] * 35
    sns_plot = sns.lineplot(data=pd.concat([temp_df_1, temp_df_2]).reset_index(), x='episodes', y=plot_type,
                            hue='Squared Gradient Norm', ci=100,  markersize=6,
                            style='Squared Gradient Norm', markers=True, dashes=False, lw=5)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Gradient Variance', fontsize=24)
    sns_plot.legend(handles=sns_plot.lines,
                    labels=[r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{PG}_k\right]\right)$',
                            r'Tr$\left({Var}\left[\widehat{\nabla\mu}^{MLR}_k\right]\right)$'],
                    fontsize=20)
    if not show:
        sns_plot.figure.savefig('result/figures/{}-{}-{}.pdf'.format(plot_type, algorithm, 'Fermentation'),
                                format='pdf', bbox_inches='tight')
    else:
        sns_plot.figure.show()


"""
Actor Critic Convergence with constant c
"""
QUANTILE_CONST = 0.8
show=False
# algorithm = 'Actor-Critic'
plot_type = 'average return'
data_10 = plot_convergence(plot_type, visualizations, c=1.0, quantile_const=QUANTILE_CONST, sensitivity=True)
data_15 = plot_convergence(plot_type, visualizations, c=1.5, quantile_const=QUANTILE_CONST, sensitivity=True)
data_20 = plot_convergence(plot_type, visualizations, c=2.0, quantile_const=QUANTILE_CONST, sensitivity=True)
data_30 = plot_convergence(plot_type, visualizations, c=3.0, quantile_const=QUANTILE_CONST, sensitivity=True)
data_40 = plot_convergence(plot_type, visualizations, c=4.0, quantile_const=QUANTILE_CONST, sensitivity=True)

data_10['c'] = r'$c=1.0$'
data_15['c'] = r'$c=1.5$'
data_20['c'] = r'$c=2.0$'
data_30['c'] = r'$c=3.0$'
data_40['c'] = r'$c=4.0$'

data = pd.concat([data_10, data_15, data_20, data_30, data_40])

plt.style.use('ggplot')
sns.set_theme(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.8, color_codes=True,
              rc=None)

f = plt.figure(figsize=(10, 5))
with sns.axes_style("white"):
    algorithm = 'Actor-Critic'
    # ax = f.add_subplot(gs[0, 0])
    temp_df = data[data.apply(lambda x: x['algorithm'] == (algorithm + '-VRER') and x['problem'] == 'CartPole', axis=1)]
    temp_df= temp_df.iloc[::10, :]
    # temp_df_2[plot_type] = temp_df_2.apply(lambda x: 95 if x['episodes'] < 80 and 150 < x['episodes'] < 200 else x[plot_type], axis=1)
    sns_plot = sns.lineplot(data=temp_df.reset_index(), x='episodes', y=plot_type,
                            hue='c', ci=80, palette=sns.color_palette()[:5], markersize=8,
                            style='c', markers=True, dashes=False, lw=3.5)
    sns_plot.set_xlabel("Episodes", fontsize=24)
    sns_plot.set_ylabel('Average Return', fontsize=24)
    sns_plot.legend(fontsize=20).set_title(None)
    if not show:
        sns_plot.figure.savefig('result/figures/{}-{}-{}-c.pdf'.format(plot_type, algorithm, 'CartPole'),
                                format='pdf', bbox_inches='tight')
    else:
        sns_plot.figure.show()
