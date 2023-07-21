# %%
import jax.numpy as jnp
import jax
from jax import random, vmap
from scipy.special import logit, expit
from jax.lib import xla_bridge
from jax import jit
import os
import tqdm
import pickle
import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# os.environ["CUDA_VISIBLE_DEVICES"]="1" 


import random as rd

import numpyro
numpyro.set_platform('gpu')
# numpyro.set_platform('cpu')
print('Jax Version:', jax.__version__) 
print('Numpyro Version: ', numpyro.__version__) 
print(jax.config.FLAGS.jax_backend_target) 
# print(jax.lib.xla_bridge.get_backend().platform) 
print(xla_bridge.get_backend().platform)
n_parallel = jax.local_device_count()
print('Number of compuation devices', n_parallel)


import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs,Predictive,HMCGibbs, SVI, Trace_ELBO, autoguide

from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect

from numpyro.distributions import constraints
from numpyro.distributions.transforms import StickBreakingTransform
from numpyro.distributions import Categorical
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.autoguide import AutoLaplaceApproximation, AutoNormal
from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect
from numpyro import handlers
from sklearn.metrics import mean_squared_error


import numpy as np
import sys
import pickle

from scipy import stats

import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

import time as time


# from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y
import scipy.sparse as sp
from typing import List, Sequence, Union
import warnings
from pynvml import *
import gc
from sklearn.metrics import r2_score

# %% 
def input_data_process(train_tensor_dir, test_tensor_dir,\
                        visual_face_dir, language_trait_dir):
    


    train_DATA = np.load(train_tensor_dir)
    # print(train_DATA)
    train_indexes, logit_train_rates = train_DATA[:, :-2].astype(int), logit(train_DATA[:, 3])
    # print(train_indexes)
    # print(train_rates)
    # print(min(logit_train_rates))
    # print(max(logit_train_rates))
    
    test_DATA = np.load(test_tensor_dir)
    # print(test_DATA)
    test_indexes, logit_test_rates = test_DATA[:, :-2].astype(int), logit(test_DATA[:, 3])
    # print(test_indexes)
    # print(test_rates)
    # print(min(logit_test_rates))
    # print(max(logit_test_rates))

    union_DATA = np.vstack((train_DATA, test_DATA))
    print("Union data shape: ", union_DATA.shape)
    union_indexes, logit_union_rates = union_DATA[:, :-2].astype(int), logit(union_DATA[:, 3])
    # print(union_indexes)
    # print(logit_union_rates)
    # print(min(logit_union_rates))
    # print(max(logit_union_rates))
    

    ### Load side feature datasets
    print('Loading side feature datasets ......')
    visual_face_vectors = np.load(visual_face_dir,allow_pickle=True)
    language_trait_vectors = np.load(language_trait_dir, allow_pickle=True)
    visual_face_vectors = jnp.array(visual_face_vectors)
    language_trait_vectors = jnp.array(language_trait_vectors)
    print('Visual features shape: ', visual_face_vectors.shape)
    print('Language features shape: ', language_trait_vectors.shape)


    return union_indexes, logit_union_rates, \
           train_indexes, logit_train_rates, \
           test_indexes, logit_test_rates, \
           visual_face_vectors, language_trait_vectors
    
# %%
def model(features, participants, stimulus, traits, latent_dimensions, rates = None):
    # Stick-breaking prior
    cultures = 4
    alpha = numpyro.sample('alpha', dist.Gamma(1, 1))
    with numpyro.plate('weights', cultures - 1):
        v = numpyro.sample('v', dist.Beta(1, 1))
    # print('v value: ', v)


    ### bias and scaling
    bias_prior_variance = numpyro.sample("bias_prior_variance", dist.Gamma(100, 10))
    with numpyro.plate('ind', 4476): # no of participants
        sb_trans = StickBreakingTransform()
        # print('v: ', v)
        # print('sb_trans: ', sb_trans(v))
        culture_id = numpyro.sample("culture_id", Categorical(sb_trans(v)))
        # print(culture_id.shape)
        # print(culture_id)

        bias = numpyro.sample('bias', dist.Normal(0, 1/bias_prior_variance))
        # scaling = numpyro.sample('Scale', dist.Normal(loc=0.98, scale=1/100))
        ###1/25
        scaling = numpyro.sample('LogScale', dist.HalfNormal(1/25))

    with numpyro.plate("data_loop", features.shape[0]):
        culture_assignment = culture_id[features[:, 0]]

    with numpyro.plate('competence_plate', cultures):
        ### define individual culture diff
        Mean_per_culture = numpyro.sample("Mean_Per_culture", dist.Normal(0.0, 1.0))
        print('Mean_per_culture: ', Mean_per_culture.shape)
        precision_per_culture = numpyro.sample('competence_precision', dist.Gamma(20.0, 2.0))
        # logItemDiffPrecision = numpyro.sample("logItemDiffPrecision", Gamma(100, 5))
        with numpyro.plate('c_compet', 4476): # number of participant
            logCompetence = numpyro.sample('logCompetence', dist.Normal(Mean_per_culture, 1/precision_per_culture))


        ###bayesian tensor factorization
        ### (6.0, 1.0)
        visual_f_prior = numpyro.sample("visual_f_prior", dist.Gamma(1.0, 0.01))
        with numpyro.plate('latent_visual_coefficients', 512):
            with numpyro.plate('visual_dimensions', 60):
                visual_feature_coefficient =  numpyro.sample("visual_feature_coefficient",dist.Normal(0, 1/visual_f_prior))
                print('visual_feature_coefficient',visual_feature_coefficient.shape)
                # print('visual_feature_coefficient',visual_feature_coefficient)
                ### (512 latent vector weights, 10 latent spaces)

        language_f_prior = numpyro.sample("language_f_prior", dist.Gamma(1.0, 0.01))
        # language_f_prior = numpyro.sample("language_f_prior", dist.Gamma(1.0, 0.01))
        # print('language_f_prior shape: ', language_f_prior.shape)
        with numpyro.plate('latent_language_coefficients', 300):
                with numpyro.plate('language_dimensions', 60):
                    language_feature_coefficient =  numpyro.sample("language_feature_coefficient", dist.Normal(0, 1/language_f_prior))
                    # print(language_feature_coefficient.reshape(300, 10))
                    print('language_feature_coefficient', language_feature_coefficient.shape)

        
        
        stimuli = 1000
        with numpyro.plate('stimplate', stimuli):
                LogItem_diff = numpyro.sample('LogItem_diff', dist.Normal(0, 1/20))

    ### fuse them togetherls
    print('visual_face_vectors shape ', visual_face_vectors.shape)
    mu_part1 = vmap(lambda vis_vec, vis_coefs:
                    jnp.dot(vis_vec, vis_coefs.T), in_axes = (0,None), out_axes = 1)(
                    visual_face_vectors,
                    visual_feature_coefficient
                    )
    # mu_part1 = jnp.einsum('ab, cdb->adb', visual_face_vectors, visual_feature_coefficient)
    print('mu_part1: ', mu_part1.shape)

    mu_part2 = vmap(lambda lang_vec, lang_coefs:
                    jnp.dot(lang_vec, lang_coefs.T), in_axes = (0,None), out_axes = 1)(
                    language_trait_vectors,
                    language_feature_coefficient
                    )
    print('mu_part2: ', mu_part2.shape)


    #### create consensus
    # mid_product = jnp.einsum('abc, adc->adb', mu_part1, mu_part2)
    mid_product = vmap(lambda mu1, mu2:
            jnp.dot(mu1, mu2.T), in_axes = (0, 0), out_axes = 0)(
            mu_part1,
            mu_part2)
    print('midproduct Shape : ', mid_product.shape)
    # print('midproduct : ', mid_product)

    consensus = numpyro.deterministic("consensus", mid_product)
    print('consensus Shape : ', consensus.shape)
    # print('consensus : ', consensus)


    # print('Shape of bias: ', bias[features[:, 0]].shape)
    # print('bias: ', bias[features[:, 0]])
    # print('Shape of scaling: ', scaling[features[:, 0]].shape)
    # print('scaling: ', scaling[features[:, 0]])

    print('-----------------------------')
    print('min of log scaling: ', jnp.min(jnp.log(scaling[features[:, 0]])))
    print('max of log scaling: ', jnp.max(jnp.log(scaling[features[:, 0]])))
    print('average of log scaling: ', jnp.average(jnp.log(scaling[features[:, 0]])))
    # print('min of scaling: ', jnp.min(scaling[features[:, 0]]))
    # print('max of scaling: ', jnp.max(scaling[features[:, 0]]))
    # print('average of scaling: ', jnp.average(scaling[features[:, 0]]))
    print('-----------------------------')
    print('min of bias: ', jnp.min(bias[features[:, 0]]))
    print('max of bias: ', jnp.max(bias[features[:, 0]]))
    print('average of bias: ', jnp.average(bias[features[:, 0]]))
    print('-----------------------------')
    print('min of consensus: ', jnp.min(consensus[culture_assignment,features[:, 1]-1, features[:, 2]-1]))
    print('max of consensus: ', jnp.max(consensus[culture_assignment,features[:, 1]-1, features[:, 2]-1]))
    print('average of consensus: ', jnp.average(consensus[culture_assignment,features[:, 1]-1, features[:, 2]-1]))
    print('-----------------------------')

 
    mu = vmap(lambda cons, b, scale: (jnp.exp(scale) * (cons)) + b)(consensus[culture_assignment,features[:, 1]-1, features[:, 2]-1],
                                                                    bias[features[:, 0]],
                                                                    scaling[features[:, 0]]
                                                                    )

    # print(' mu_0: ', mu_0)

    # mu = mu_0[jnp.arange(len(mu_0)), culture_assignment]
    print('-----------------------------')
    # print(culture_assignment)
    # print(' mu: ', mu)
    print('min of mu: ', jnp.min(mu))
    print('max of mu: ', jnp.max(mu))
    print('average of mu: ', jnp.average(mu))
    # print('min of rates: ', jnp.min(rates))
    # print('max of rates: ', jnp.max(rates))
    # print('average of rates: ', jnp.average(rates))
    print('-----------------------------')

    competence = jnp.exp(logCompetence[features[:, 0],culture_assignment])
    # print('culture_assignment: ', culture_assignment)
    print('logCompetence shape: ', logCompetence.shape)
    # print('logCompetence: ', logCompetence)
    # competence = jnp.exp(logCompetence[features[:, 0], culture_assignment])
    print('competence shape: ', competence.shape)
    # print('competence: ', competence)


    itemDifficulty = jnp.exp(LogItem_diff[features[:, 1]-1, culture_assignment])
    print('itemDifficulty shape: ', itemDifficulty.shape)
    scale_var = jnp.exp(scaling[features[:, 0]])
    print('scale_var shape: ', scale_var.shape)
    # ratingVariance = (competence * itemDifficulty * scale)**2
    ratingVariance = vmap(lambda comp, diff, scale: ((scale * comp * diff)**2))(competence,
                                                                                itemDifficulty,
                                                                                scale_var
                                                                                )

    # ratingVariance = jnp.mean(ratingVariance_0, axis = 1)
    print('Shape of ratingVariance : ', ratingVariance.shape)
    # print('ratingVariance : ', ratingVariance)
    # ratingVariance = (competence * scale)**2
    print('-----------------------------')
    print('min of ratingVariance: ', jnp.min(ratingVariance))
    print('max of ratingVariance: ', jnp.max(ratingVariance))
    print('average of ratingVariance: ', jnp.average(ratingVariance))
    print('-----------------------------')

    # sigma = numpyro.sample("sigma_val", dist.HalfNormal(4.0))
    # print('sigma val: ', sigma.shape)
    # with numpyro.plate("data_out", features.shape[0]):
    # with numpyro.plate("item", 1, dim=-1):
    numpyro.sample("rating", dist.Normal(mu, ratingVariance), obs = rates)
  

# %%
def predict_fn_jit(rng_key, samples, *args, **kwargs):
    return Predictive(model, samples, parallel=True)(rng_key, *args, **kwargs)

# %%

def prediction_outtable_df(data_index, logit_data_rates, pred_ratings, pred_ratings_std):

    participant_lst = list(data_index[:, 0]) ### stimulus list -- no unique
    # print(len(participant_lst))
    stimulus_lst = list(data_index[:, 1]) ### stimulus list -- no unique
    # print(len(stimulus_lst))
    trait_lst = list(data_index[:, 2]) ### trait list -- no unqiue
    # print(len(trait_lst))
    rating_lst = list(expit(logit_data_rates) * 100) ### trait list -- no unqiue
    # print(len(rating_lst))

    prediction_out_df = pd.DataFrame(columns=['participant', 'stimulus', 'trait', 'rating_true', 'rating_pred', 'rating_std'])
    prediction_out_df['participant'] = participant_lst
    prediction_out_df['stimulus'] = stimulus_lst
    prediction_out_df['trait'] = trait_lst
    prediction_out_df['rating_true'] = rating_lst
    prediction_out_df['rating_pred'] = pred_ratings
    prediction_out_df['rating_std'] = pred_ratings_std

    prediction_out_df['rating_pred_inf'] = prediction_out_df['rating_pred'] - prediction_out_df['rating_std']
    prediction_out_df['rating_pred_sup'] = prediction_out_df['rating_pred'] + prediction_out_df['rating_std']

    print(prediction_out_df.shape)
    print(prediction_out_df.head(5))
    print(prediction_out_df.describe())

    return prediction_out_df
# %%
def RMSE_val(pred_ratings, actual_ratings):
    # RMSE_value = mean_squared_error(prediction_out_df['rating_true'], prediction_out_df['rating_pred'], squared=False)
    RMSE_value = mean_squared_error(pred_ratings, actual_ratings, squared=False)
    return RMSE_value
# %%
# %%
def R2_val(pred_ratings, actual_ratings):
    R2_value = r2_score(pred_ratings, actual_ratings)
    return R2_value
# %%

# # %%
def lst_flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in lst_flatten(item):
                 yield x
         else:        
             yield item
# # %%
# %%
if __name__ == '__main__': 


    ### read in training and test data as well as side features
    train_tensor_dir = "training_data_all_95.npy"
    test_tensor_dir = "test_data_all_05.npy"

    ##########################
    visual_face_dir = "face_vectors.npy"
    language_trait_dir = "language_vectors.npy"

    union_indexes, logit_union_rates,\
    train_indexes, logit_train_rates, \
    test_indexes, logit_test_rates, \
    visual_face_vectors, language_trait_vectors = input_data_process(train_tensor_dir, 
                                                                     test_tensor_dir, 
                                                                     visual_face_dir, 
                                                                     language_trait_dir)
    # print('all data indexes: -------')
    # print(union_indexes.dtype)
    # print(union_indexes.shape)
    print('#############################')



    all_participants = len(np.unique(union_indexes[:, 0]))  # number of participants (was K)
    print('no. of all participants: ', str(all_participants))
    all_stimulus = len(np.unique(union_indexes[:, 1]))  # number of stimuli (was N)
    print('no. of all stimulus: ', str(all_stimulus))
    all_traits = len(np.unique(union_indexes[:, 2]))  # number of traits (was T)
    print('no. of all traits: ', str(all_traits))
    all_responses = len(logit_union_rates)  # number of responses (was R)
    print('no. of all responses: ', str(all_responses))


    ### Process training data
    print('training data indexes: -------')
    # print(train_indexes.dtype)
    print(train_indexes.shape)
    print('logit_train_rates: -------')
    # print(expit(logit_train_rates))
    print('logit train rates min: ', np.min(logit_train_rates))
    print('logit train rates max: ', np.max(logit_train_rates))

    #################################################
    #################################################

    #### Process test data
    ### import test data
    print('test indexes: -------')
    # print(test_indexes.dtype)
    # print(test_indexes)
    print(test_indexes.shape)
    print('logit_test_rates: -------')
    # print(DATA_test[:, 3] )
    # print(expit(logit_test_rates))
    print('logit test rates min: ', np.min(logit_test_rates))
    print('logit test rates max: ', np.max(logit_test_rates))

    #################################################
    #################################################



    latent_dim = 60  # The dimensionality of the latent space (was DIMENSIONS)
    no_cultures = 4
    ### if choose mcmc simulation
    print('********************* Simulation begins !!!!!!!!! *********************')
    kernel = DiscreteHMCGibbs(NUTS(model, init_strategy=numpyro.infer.init_to_sample, step_size=1e-3,target_accept_prob= 0.80, max_tree_depth=8))
    mcmc = MCMC(kernel, num_warmup=50, num_samples=3000, chain_method='vectorized', num_chains= 1, progress_bar=True, jit_model_args=True)
 
    

    mcmc.run(random.PRNGKey(0),features = train_indexes, 
            participants = all_participants,
            stimulus= all_stimulus,
            traits= all_traits, 
            latent_dimensions = latent_dim, 
            rates= logit_train_rates)


    posterior_samples = mcmc.get_samples()

    print('Diagnos stats !!!! --------------------------------')
    # print(mcmc.print_summary())
    diagnos = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))

    with open('v6FusionTest_sm30_3000_latent60_culture4_95vs5.pickle', 'wb') as handle:
        pickle.dump(diagnos, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save('v6FusionTest_sm30_3000_latent60_culture4_95vs5.npy', posterior_samples['culture_id'])

    ##### compute rhat percentage
    loaded_dictionary = diagnos
    print(loaded_dictionary.keys())

    print(type(loaded_dictionary['visual_feature_coefficient']))
    print(loaded_dictionary['visual_feature_coefficient'].keys())

    # for para in loaded_dictionary.keys():
    for para in ['LogItem_diff', 'bias', 'LogScale', 'language_feature_coefficient', 'logCompetence', 'visual_feature_coefficient']:
        print('-----------------------', para)
        print(loaded_dictionary[para]['r_hat'].shape)
        rhat_lst = loaded_dictionary[para]['r_hat'].tolist()
        rhat_lst_flat = list(lst_flatten(rhat_lst))
        print('no of parameters: ', str(len(rhat_lst_flat)))
        counter0 = 0
        counter1 = 0
        counter2 = 0
        for e in rhat_lst_flat:
            # print(e)
            if e <= 1.1:
                counter0 = counter0 + 1
            if e <= 1.3:
                counter1 = counter1 + 1
            if e <= 1.5:
                counter2 = counter2 + 1
                # print(counter)
        print('good rhat < 1.1 % ', str(counter0/ len(rhat_lst_flat)))
        print('good rhat < 1.3 % ', str(counter1/ len(rhat_lst_flat)))
        print('good rhat < 1.5 % ', str(counter2/ len(rhat_lst_flat)))


    print('end 1-----------------------')

    print('Yangyang approach --------------------------------')
    print('posterior samples type: ' , type(posterior_samples))
    with open('v6posterior_sm30_3000_latent60_culture4_95vs5.pickle', 'wb') as handle:
        pickle.dump(posterior_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('end 2-----------------------')

    latent_dim = 60  # The dimensionality of the latent space (was DIMENSIONS)
    no_cultures = 4

    # predictions_jit = predict_fn_jit(random.PRNGKey(0), posterior_samples, 
    #                                 features = train_indexes, 
    #                                 participants = all_participants,
    #                                 stimulus= all_stimulus,
    #                                 traits= all_traits, 
    #                                 latent_dimensions = latent_dim)["rating"]
    test_predictions_jit = predict_fn_jit(random.PRNGKey(0), posterior_samples, 
                                            features = test_indexes,
                                            participants = all_participants,
                                            stimulus = all_stimulus,
                                            traits = all_traits,
                                            latent_dimensions =latent_dim)["rating"]

    # ## compute training prediction
    # model_pred_rating_array = np.array(predictions_jit)
    # print('Shape of original training predicted rating arrays', model_pred_rating_array.shape)
    # # print('Type of original predicted rating arrays', type(model_pred_rating_array))

    # pred_ratings = np.round(np.nanmean((expit(model_pred_rating_array) * 100),axis = 0), decimals = 4)
    # # print('Shape of training predicted rating values', pred_ratings.shape)
    # # print('Training predicted rating values', pred_ratings)
    # pred_ratings_std = np.round(np.nanstd((expit(model_pred_rating_array) * 100),axis = 0), decimals = 4)
    # # print('Shape of training predicted rating std', pred_ratings_std.shape)

    ### compute test prediction
    test_model_pred_rating_array = np.array(test_predictions_jit)

    # print('Shape of original test predicted rating arrays', test_model_pred_rating_array.shape)
    test_pred_ratings = np.round(np.nanmean((expit(test_model_pred_rating_array) * 100),axis = 0), decimals = 4)
    # test_pred_ratings[np.isnan(test_pred_ratings)] = 0
    # print('Shape of test predicted rating values', test_pred_ratings.shape)
    # print('Test predicted rating values', test_pred_ratings)
    # print('Type of Test predicted rating values', type(test_pred_ratings))
    test_pred_ratings_std = np.round(np.nanstd((expit(test_model_pred_rating_array) * 100),axis = 0), decimals = 4)
    test_pred_ratings_std[np.isnan(test_pred_ratings_std)] = 0
    print('Shape of test predicted rating std', test_pred_ratings_std.shape)
    # del predictions_jit, test_predictions_jit, posterior_samples
    del test_predictions_jit, posterior_samples

    # ## output training predictions
    # print('Initial Training Prediction: ................')    
    # train_prediction_out_df = prediction_outtable_df(train_indexes, logit_train_rates, pred_ratings, pred_ratings_std)

    # train_RMSE = RMSE_val(pred_ratings, expit(logit_train_rates) * 100.0)
    # train_R2 = R2_value(pred_ratings, expit(logit_train_rates) * 100.0)
    # print('*************RMSE of initial train: ', train_RMSE)
    # print('*************R2 value of initial train: ', train_R2)
    #######################################
    #######################################
    ## output test predictions
    print('Initial Test Prediction: ................')
    test_prediction_out_df = prediction_outtable_df(test_indexes, logit_test_rates, test_pred_ratings, test_pred_ratings_std)

    test_RMSE = RMSE_val(test_pred_ratings, expit(logit_test_rates) * 100.0)
    print('*************RMSE of initial test: ', test_RMSE) 
    test_R2 = R2_val(test_pred_ratings, expit(logit_test_rates) * 100.0)  
    print('*************R2_value of initial test: ', test_R2)        




    ######################################
    ######################################
 
  