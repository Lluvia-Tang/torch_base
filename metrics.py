import torch
from sklearn import metrics
from loss import loss_function, Bandit_loss_function
import numpy as np

np.set_printoptions(threshold=np.inf)

def get_acc_from_logits(logits, labels):
    soft_probs = torch.argmax(logits, 1)
    acc = (soft_probs == labels).float().mean()
    return acc, soft_probs, labels

inputs_cols = ['text_indices','attention_mask','target_indices','tag_indices','position_indices',"graph"]
# inputs_cols = ['text_indices','attention_mask','target_indices',"graph"]  #bert时


def evaluate_MT(model, dataloader, device): #task_type
    model.eval()
    mean_acc, mean_loss, count = 0, 0.0, 0
    st_n_pre, st_n_labels = None, None
    senti_n_pre, senti_n_labels = None, None
    with torch.no_grad():
        for i, sample_batched in enumerate(dataloader):
            x = [sample_batched[col].cuda() for col in inputs_cols]
            stance = sample_batched['stance'].to(device)
            sentiment = sample_batched['sentiment'].to(device)
            st_pred, senti_pred, stance_weight, sentiment_weight = model(x)
            # st_pred = model(x)
            # senti_pred = model(x)
            # stance_weight = sentiment_weight = None

            # calculate loss
            loss, (stance_loss, sentiment_loss) = loss_function(
                stance_predict=st_pred,
                stance_target=stance,
                sentiment_predict=senti_pred,
                sentiment_target=sentiment,
                # lexicon_vector=train_lexicon,
                stance_weight=stance_weight,
                sentiment_weight=sentiment_weight,
                stance_loss_weight=0.7)
            # lexicon_loss_weight=config.lexicon_loss_weight,)

            mean_loss += loss.item()
            # stance_loss += batch_stance_loss
            # sentiment_loss += batch_sentiment_loss
            # lexicon_loss += batch_lexicon_loss

            if (st_n_pre == None):
                st_n_pre = torch.argmax(st_pred, 1)
                st_n_labels = stance
                senti_n_pre = torch.argmax(senti_pred, 1)
                senti_n_labels = sentiment
            else:
                st_n_pre = torch.cat((st_n_pre, torch.argmax(st_pred, 1)), dim=0)
                st_n_labels = torch.cat((st_n_labels, stance), dim=0)
                senti_n_pre = torch.cat((senti_n_pre, torch.argmax(senti_pred, 1)), dim=0)
                senti_n_labels = torch.cat((senti_n_labels, sentiment), dim=0)
            count += 1
    # print(n_pre.cpu())
    # print(n_labels.item())
    # print(n_pre.item())
    acc = metrics.accuracy_score(st_n_labels.cpu(), st_n_pre.cpu())
    f1 = metrics.f1_score(st_n_labels.cpu(), st_n_pre.cpu(), labels=[0, 1], average='macro')
    mif1 = metrics.f1_score(st_n_labels.cpu(), st_n_pre.cpu(), labels=[0, 1], average='micro')
    re = metrics.recall_score(st_n_labels.cpu(), st_n_pre.cpu(), labels=[0, 1], average='macro')
    pr = metrics.precision_score(st_n_labels.cpu(), st_n_pre.cpu(), labels=[0, 1], average="macro")
    f1_favor = metrics.f1_score(st_n_labels.cpu(), st_n_pre.cpu(), labels=[1], average='macro')
    f1_against = metrics.f1_score(st_n_labels.cpu(), st_n_pre.cpu(), labels=[0], average='macro')

    senti_acc = metrics.accuracy_score(senti_n_labels.cpu(), senti_n_pre.cpu())
    senti_f1 = metrics.f1_score(senti_n_labels.cpu(), senti_n_pre.cpu(), labels=[0, 1], average='macro')
    senti_mif1 = metrics.f1_score(senti_n_labels.cpu(), senti_n_pre.cpu(), labels=[0, 1], average='micro')
    senti_re = metrics.recall_score(senti_n_labels.cpu(), senti_n_pre.cpu(), labels=[0, 1], average='macro')
    senti_pr = metrics.precision_score(senti_n_labels.cpu(), senti_n_pre.cpu(), labels=[0, 1], average="macro")

    return acc, mean_loss / count, pr, re, f1, mif1,senti_acc, senti_f1, senti_mif1,senti_re, senti_pr, f1_favor, f1_against

def step(model, sample_batched):
    # img, label = data
    # # warp input打包数据
    # img = Variable(img).cuda()
    # label = Variable(label).cuda()

    x = [sample_batched[col].cuda() for col in inputs_cols]
    label = sample_batched['stance'].cuda()
    # label = sample_batched['sentiment'].cuda()

    # compute output
    # pred, attention = model(x)
    pred = model(x)
    return x, label, pred

def evaluate(model, criterion, dataloader, device): #task_type
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0
    n_pre, n_labels = None, None
    with torch.no_grad():
        for i, sample_batched in enumerate(dataloader):
        # x, label, pred = self.step(data)
        #     metrics = self.compute_metrics(pred, label, is_train=False)
        # for input_ids, attention_mask, types, labels, graphs in tqdm(dataloader, desc="Evaluating"):
        #     input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        #     logits = model(input_ids, attention_mask)
            x, label, pred = step(model,sample_batched)
            # labels = labels.permute(1, 0)
            # labels = labels[task_type]
            mean_loss += criterion(pred, label).item()
            # print("+++++mean_loss: ",mean_loss)
            # print("+++++++pred :",pred)
            # print("+++++++label :",label)
            k, y, l = get_acc_from_logits(pred, label)
            # print(k,"\t",y,"\t",l)

            mean_acc += k
            if (n_pre == None):
                n_pre = y
                n_labels = l
            else:
                n_pre = torch.cat((n_pre, y), dim=0)
                n_labels = torch.cat((n_labels, l), dim=0)
            count += 1
    # print(n_pre.cpu())
    # print(n_labels.item())
    # print(n_pre.item())
    # print("set(y_test) - set(y_pred) = ",set(n_labels.cpu()) - set(n_pre.cpu()))
    # print(n_labels.cpu())
    # print(n_pre.cpu().numpy())
    acc = metrics.accuracy_score(n_labels.cpu(), n_pre.cpu())
    f1 = metrics.f1_score(n_labels.cpu(), n_pre.cpu(), labels=[0, 1], average='macro')
    mif1 = metrics.f1_score(n_labels.cpu(), n_pre.cpu(), labels=[0, 1], average='micro')
    re = metrics.recall_score(n_labels.cpu(), n_pre.cpu(), labels=[0, 1], average='macro')
    pr = metrics.precision_score(n_labels.cpu(), n_pre.cpu(), labels=[0, 1], average="macro")
    f1_favor = metrics.f1_score(n_labels.cpu(), n_pre.cpu(), labels=[1], average='macro')
    f1_against = metrics.f1_score(n_labels.cpu(), n_pre.cpu(), labels=[0], average='macro')
    return acc, mean_loss / count, pr, re, f1, mif1, f1_favor, f1_against