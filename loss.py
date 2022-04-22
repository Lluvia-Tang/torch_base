# 3rd-party module
import math

import torch.nn as nn
import torch.nn.functional as F

# def loss_function(stance_predict, stance_target,
#                   sentiment_predict, sentiment_target,
#                   lexicon_vector,
#                   stance_weight, sentiment_weight,
#                   stance_loss_weight, lexicon_loss_weight,
#                   ignore_label):
def loss_function(stance_predict, stance_target,
                  sentiment_predict, sentiment_target,
                  stance_weight, sentiment_weight,
                  stance_loss_weight):

    # # get cross entropy loss
    stance_loss = F.cross_entropy(stance_predict, stance_target, ignore_index=2)
    sentiment_loss = F.cross_entropy(sentiment_predict, sentiment_target, ignore_index=2)

    # stance_loss = F.cross_entropy(stance_predict, stance_target)
    # sentiment_loss = F.cross_entropy(sentiment_predict, sentiment_target)

    # get attention weight
    sum_weight = stance_weight + sentiment_weight
    norm_weight = sum_weight / sum_weight.max(dim=1, keepdim=True)[0]

    # get MSE loss (lexicon loss)
    #lexicon_loss = F.mse_loss(norm_weight, lexicon_vector)

    # get final loss
    total_loss = (
        # stance_loss + 0
        # + sentiment_loss
        stance_loss_weight * stance_loss
        + (1-stance_loss_weight) * sentiment_loss)
    #     + lexicon_loss_weight * lexicon_loss
    # )

    return total_loss, (stance_loss, sentiment_loss) #, lexicon_loss)

def Bandit_loss_function(stance_predict, stance_target,
                  sentiment_predict, sentiment_target,
                  stance_weight, sentiment_weight,
                  stance_loss_weight,p_k):

    # # get cross entropy loss
    stance_loss = F.cross_entropy(stance_predict, stance_target, ignore_index=2)
    sentiment_loss = F.cross_entropy(sentiment_predict, sentiment_target, ignore_index=2)

    # stance_loss = F.cross_entropy(stance_predict, stance_target)
    # sentiment_loss = F.cross_entropy(sentiment_predict, sentiment_target)

    # get attention weight
    # sum_weight = stance_weight + sentiment_weight
    # norm_weight = sum_weight / sum_weight.max(dim=1, keepdim=True)[0]

    # get MSE loss (lexicon loss)
    #lexicon_loss = F.mse_loss(norm_weight, lexicon_vector)

    # get final loss
    total_loss = (
        p_k[0] * stance_loss
        + p_k[1] * sentiment_loss)
    #     + lexicon_loss_weight * lexicon_loss
    # )
    #     loss_t = criterion(y_, y)
    #     losses.append(loss_t.item())
    #     if i > 0:
    #         loss += loss_t * p_k[i]
    #     else:
    #         loss = loss_t * p_k[i]


    # loss = loss / len(tasks)
    # opt_g.zero_grad()
    # opt_c.zero_grad()
    # loss.backward()
    # opt_g.step()
    # opt_c.step()


    return total_loss, (stance_loss, sentiment_loss) #, lexicon_loss)

