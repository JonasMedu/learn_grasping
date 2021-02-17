"""
File to create visualisation graphs for the corresponding scientific work.
"""

import pandas as pd

from performance_analysis.performanceVisual import make_policy_evaluation_noise
from performance_analysis.utils import reorderLegend
from setting_utils.param_handler import TB_DIR
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import product
run_names = os.listdir(TB_DIR)
config_file = "/config"

pt = 0.0138  # inch. https://www.overleaf.com/learn/latex/Lengths_in_LaTeX
textwidth_in_inches = 418.25554 * pt

font = {'family': 'normal',
        'color':  'black',
        'weight': 'normal',
        'size': 11,
}

class MakingGraphs(object):
    def __init__(self, path_dir):
        self.path_dir = path_dir
        print('init')

    @staticmethod
    def get_progress_array(path_dir, name):
        try:
            params = pd.read_csv(os.path.join(path_dir, 'config'))
            progress = pd.read_csv(os.path.join(path_dir, 'progress.csv'))
            params_ser = pd.Series(dict(zip(params.iloc[:,0], params.iloc[:,1])))
            return {'name': name, 'progress': progress, 'params':params_ser}
        except Exception as et:
            print(et)
            print('name', name)
            pass

    @staticmethod
    def get_column_id_dict(df_list: list):
        # get column dict
        assert isinstance(df_list[0], pd.DataFrame)
        columns = df_list[0].columns
        return dict(zip(columns, np.arange(len(columns))))

    @staticmethod
    def get_dfs_for_run_tag(min_number_runs=4):
        """returns a list of data frames from the progress csv"""
        only_tags = set([name[:-61] for name in run_names])
        tag_dfs = {}
        for tag in only_tags:
            tags_list = list(filter(lambda x: tag == x[:-61], run_names))
            ser_list = [MakingGraphs.get_progress_array(os.path.join(TB_DIR, name), name) for name in tags_list]
            ser_list = list(filter(lambda s: s is not None, ser_list))
            if len(ser_list) > min_number_runs:
                progesses = [ser['progress'] for ser in ser_list]
                filled_df = MakingGraphs.merge_progress_dfs(progesses)
                tag_dfs.update({tag:filled_df})
        return tag_dfs

    @staticmethod
    def merge_progress_dfs(list_of_dfs):
        num_trj_df = pd.concat([prog.loc[:, 'num_trajectories'] for prog in list_of_dfs], axis=1)
        return num_trj_df.fillna(axis=0, method='ffill')


def get_colors(labels: set):
    assert len(labels) < 25, "Can not produce such a high number of unique labels"
    sns.reset_orig()  # get default matplotlib styles back
    #clrs = sns.color_palette('muted', n_colors=10)  # a list of RGB tuples
    clrs = sns.color_palette("ch:s=-.2,r=.6")# , n_colors=len(labels))
    LINE_STYLES = ['solid', 'dotted', 'dashdot']
    return dict(zip(labels, product(LINE_STYLES, clrs)))


def grouped_progress_image(rolling=5, keep_names: dict = None, save_name='Training_progress'):
    dit = MakingGraphs.get_dfs_for_run_tag(min_number_runs=4)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(textwidth_in_inches, textwidth_in_inches * 2/3)
    if keep_names:
        dit_k = set(dit.keys()).intersection(set(keep_names.keys()))
        dit = dict(zip(dit_k, [dit[name] for name in dit_k]))
    cmap = get_colors(set(dit.keys()))
    for name in dit:
        linestyle, color = cmap[name]
        array = dit[name]
        m = array.mean(axis=1)
        m_roll = m.rolling(rolling).mean()[rolling-1:]
        m = m[rolling-1:]
        t = range(0, len(m))
        ax.plot(t, m, lw=1, label=keep_names[name], color=color, linestyle=linestyle, alpha=1.)
        ax.plot(t, m_roll, lw=1, color=color, linestyle=linestyle)
        # sd = np.log(array.std(axis=1))
        #ax.fill_between(t, m+sd, np.clip(m-sd, 0, np.inf), alpha=0.5, label='95% intvl.', color=cmap[name])

    reorderLegend(ax)

    ax.set_yticks(np.linspace(0, 3000, 10))
    ax.set_yticklabels(np.linspace(0, 3000, 10))
    ax.set_xticks([100, 400, 800])
    ax.set_xticklabels([100, 400, 800])

    plt.yscale('log')
    ax.set_ylabel('Number of trajectories per run in log scale')
    ax.set_xlabel('Training iteration')
    ax.set_title('Learning progression')
    plt.tight_layout()
    fig.savefig(save_name + '.png', dpi=300)
    #plt.show()


class NoiseGaph(object):
    my_cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    @staticmethod
    def get_outcome_for_run_tag(min_number_runs=4):
        """returns a list of data frames from the progress csv"""
        only_tags = set([name[:-61] for name in run_names])
        tag_dfs = {}
        tag_configs = {}
        for tag in only_tags:
            tags_list = list(filter(lambda x: tag == x[:-61], run_names))
            ser_list = [MakingGraphs.get_progress_array(os.path.join(TB_DIR, name), name) for name in tags_list]
            ser_list = list(filter(lambda s: s is not None, ser_list))
            if len(ser_list) > min_number_runs:
                config_df = pd.concat([ser['params'] for ser in ser_list], axis=1)
                progesses = [ser['progress'] for ser in ser_list]
                filled_df = MakingGraphs.merge_progress_dfs(progesses)
                tag_dfs.update({tag:filled_df})
                tag_configs.update({tag:config_df})
        return tag_dfs, tag_configs

    def __init__(self):
        dit, config_it = self.get_outcome_for_run_tag(min_number_runs=4)
        self.noise_dfs = dit['NoisyEnv']
        noise_config = config_it['NoisyEnv']
        # (indices of columns and runs match)
        self.noise_dfs.columns = noise_config.loc['run_name', :]
        noise_index = pd.Series(index=noise_config.loc['run_name', :], data=list(noise_config.loc['movement_noise', :]))
        noise_index = noise_index.astype(float)
        self.cmap = self.getCmap(noise_index.sort_values().index)
        sn_df = NoiseGaph.create_noise_score_df(self.cmap, self.noise_dfs, noise_index, 20)
        self.noise_score_df = sn_df
        self.noise_index = noise_index


    @staticmethod
    def compress_results(tag='Middle'):
        """"QUick debug method"""
        noMiddle_names = [name for name in run_names if name.find(tag)>-1]
        ser_list = [MakingGraphs.get_progress_array(os.path.join(TB_DIR, name), name) for name in noMiddle_names]
        # filter for malicious runs.
        ser_list = [ser for ser in ser_list if ser]
        # construct data frames
        for exp_run in ser_list:
            # 1. save progress as csv with run_name as name
            df = pd.DataFrame(exp_run['progress'])
            df.to_csv(exp_run['name'])
        names = [run['name'] for run in ser_list]
        noises = [run['params']['movement_noise'] for run in ser_list]
        pd_series = pd.Series(data=noises, index=names)
        # pd_series.to_csv("noisy_run_meta")
        return pd_series

    @staticmethod
    def getCmap(labels):
        assert len(labels) < 25, "Can not produce such a high number of unique labels"
        sns.reset_orig()  # get default matplotlib styles back
        # colors =  sns.light_palette('black', n_colors=len(labels))
        colors = [NoiseGaph.my_cmap(c) for c in np.linspace(0, 1, len(labels))]
        return dict(zip(labels, colors))

    @staticmethod
    def plot_line_fig(noise_index: pd.Series, cmap: dict, df_list_map: pd.DataFrame, rolling: int = 20):
        fig, ax = plt.subplots(1)
        fig.set_size_inches(textwidth_in_inches, textwidth_in_inches * 2/3)
        fig.set_tight_layout(True)

        for name, val in noise_index.sort_values().items():
            nn = np.round(val, 2)
            color = cmap[name]
            array = df_list_map.loc[:, name]
            m_roll = array.rolling(rolling).mean()[rolling - 1:]
            t = range(0, len(array[rolling - 1:]))
            ax.plot(t, array[rolling - 1:], lw=1, label=nn, color=color)

        # alternative legend
        # cbar = fig.colorbar(hexbins, cax=axins1, orientation='horizontal', ticks=[below, above])
        sm = plt.cm.ScalarMappable(cmap=NoiseGaph.my_cmap, norm=plt.Normalize(vmin=noise_index.min(), vmax=noise_index.max()))
        plt.colorbar(sm)
        #plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        plt.margins(x=0.01)
        # fixed for paper
        ax.set_yticks([64, 150, 500, 1000])
        ax.set_yticklabels([64, 150, 500, 1000])
        ax.set_xticks([100, 400])
        ax.set_xticklabels([100, 400])
        # /fixed for paper
        ax.set_ylabel('Number of trajectories per run')
        ax.set_xlabel('Training iteration')
        # ax.set_title('Learning progression') # in paper
        fig.savefig('Training_progress_noisy_run' + '.png', bbox_inches='tight', dpi=300)

    @staticmethod
    def plot_noise_score_fig(noise_score: pd.DataFrame):
        fig, ax = plt.subplots(1)
        fig_width = textwidth_in_inches * .49
        fig.set_size_inches(fig_width, fig_width * 2/3)
        ax.scatter(noise_score['noise'], noise_score['num_trajectories'], c=noise_score['color'], s=10,
                    marker='o')

        plt.margins(x=0.01)
        x_tics = ax.get_xticks()
        plt.hlines(64, xmin=min(x_tics), xmax=max(x_tics), linestyles='--', color='grey', linewidth=1)
        ax.set_yticks([64, 200, 800])  # 3200/50 = 64 : Score for trajectory length of 50
        ax.set_yticklabels([64, 200, 800])

        plt.xscale('log')
        ax.set_ylabel('Mean trajectories')
        ax.set_xlabel('Noise magnitude')
        ax.set_xticks(np.round(np.geomspace(1e-2, 20, 3), 2))
        ax.set_xticklabels(np.round(np.geomspace(1e-2, 20, 3), 2))

        plt.tight_layout()
        fig.savefig('performance_with_noise' + '.png', dpi=300)

    @staticmethod
    def create_noise_score_df(cmap: dict, df_list_map: pd.DataFrame, noise_index: pd.Series, rolling: int = 20):
        mean_roll_end_value_dict = {}
        for name, val in noise_index.sort_values().items():
            nn = np.round(val, 2)
            color = cmap[name]
            array = df_list_map.loc[:, name]
            m_roll = array.rolling(rolling).mean()[rolling - 1:]
            mean_roll_end_value_dict.update({name: [nn, color, m_roll.iloc[-1]]})
        sn_df = pd.DataFrame(mean_roll_end_value_dict).T
        sn_df.columns = ['noise', 'color', 'num_trajectories']
        sn_df.index.name = 'run_name'
        return sn_df

    @staticmethod
    def noise_to_action_ratio(from_file=None):
        if from_file:
            df = pd.read_csv(
                from_file)
        else:
            df = make_policy_evaluation_noise(pol_tag='NoisyEnv_', take_photos=False)
        means = df.groupby('name').mean()
        cmap = NoiseGaph.getCmap(means.sort_values('noise').index)
        c = dict(zip(list(means['noise'].values),
                     [cmap[name] for name in list(means.index)]))
        ratio_ser = means['std'] #/ means['noise'] * 10
        ratio_ser.index = np.round(means.noise, 2)
        #ratio_ser = ratio_ser.apply(np.log)
        fig, ax = plt.subplots(1)
        fig_width = textwidth_in_inches * .49
        fig.set_size_inches(fig_width, fig_width * 2/3)

        ratio_ser.sort_index().plot(kind='bar', ax=ax,
                                    color=list(pd.Series(c).sort_index().values),
                                    rot=0)
        # equilibrium line noise_{magnitude}= exp(action_{potential})+exp(10)
        x_axis_ticks = ax.get_xticks()[::5]#
        x_axis_labels = ax.get_xticklabels()[::5]
        ax.set_xticks(x_axis_ticks)
        ax.set_xticklabels(x_axis_labels)
        ax.set_xlabel('Noise magnitude')
        ax.set_ylabel('Action ratio')
        plt.margins(x=0.01)
        plt.tight_layout()
        fig.savefig('noise_ratio_bar' + '.png', dpi=300)

        fig, ax = plt.subplots(1)
        fig.set_size_inches(fig_width, fig_width * 2/3)

        plt.scatter(means['noise'], means['std'], c=pd.Series(c).sort_index(), s=10,
                    marker='o')
        ax.set_xlabel('Noise magnitude')
        ax.set_ylabel('Action ratio')
        plt.margins(x=0.05)
        plt.tight_layout()
        fig.savefig('noise_ratio_scatter' + '.png', dpi=300)
        print('Done')



progress_experimental_functions = {
    'weighted_close_to_init_x_pos_or':           'A',#, State includes object position',  # experimental graph
    'weighted_close_to_init_discount':           'B', #Joint distance (.95 discount)',  # experimental graph
    'move_connection_to_object_weighted':        'C', # Reach objects target configuration', # experimental graph
    #'movement_max_traj_length': 'Maximize trajectory length',  #  training period not long enough
    'NoGravity_diverged_Illegal':                'D', #Learn without gravity', # experimental groph
    'weighted_close_to_init_only_easy_pos_only': 'E', #Only learn one position', #exprimental groph
}

progress_experimental_functions_NAMES = {
    'A': 'State includes object position',  # experimental graph
    'B': 'Joint distance (.95 discount)',  # experimental graph
    'C': 'Reach objects target configuration', # experimental graph
    'D': 'Learn without gravity', # experimental groph  # rename
    'E': 'Only learn one position', #exprimental groph
}

progress_from_shown_reward_functions = {
    'Testing_pos23More_weighted_or_dist': 'A', #'Object orientation, sample weighted', #yes
    'weighted_or_dist': 'B', #'Object orientation', # yes
    'weighted_or_dist_discount': 'C', #'Object orientation (.95 discount)', # yes
    #'noise_weighted_or_dist': 'Object orientation (noisy), sample weighted',  # yes, rename to or_dist (TESTING refers to learning length)
    'move_connection_to_object': 'D', #'Move object to target config', #yes
    'weighted_close_to_init': 'E', #'Object position', # yes
    'close_to_ini': 'F', #'Joint position (angular similarity)', #yes
    'For_paper_close_to_init_no_debu': 'G', #'Object position simple fell definition', # yes
}
progress_from_shown_reward_functions_NAMES = {
    'A': 'Object orientation, sample weighted', #yes, # rename
    'B': 'Object orientation', # yes
    'C': 'Object orientation (.95 discount)', # yes
    'D': 'Move object to target config', #yes
    'E': 'Object position', # yes
    'F': 'Joint position (angular similarity)', #yes
    'G': 'Object position simple fell definition', # yes
}


if __name__ == '__main__':
    #print_no(tag='NoisyEnv_')
    grouped_progress_image(rolling=20, keep_names=progress_from_shown_reward_functions,
                           save_name='progress_from_shown_reward_functions')
    grouped_progress_image(rolling=20, keep_names=progress_experimental_functions,
                           save_name='progress_experimental_functions')
    ng = NoiseGaph()
    ng.plot_noise_score_fig(ng.noise_score_df)
    ng.plot_line_fig(ng.noise_index, ng.cmap, ng.noise_dfs, 20)
    # NoiseGaph.noise_to_action_ratio(from_file='/home/jhonny/Documents/imtim/images/policy_sample_positions_noise/Noise_action_potentials')
