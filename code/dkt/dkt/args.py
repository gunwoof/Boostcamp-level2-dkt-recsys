import argparse

# [찬우] str2bool 함수 추가
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    # python script 실행할 때 인자를 수정하여 실행할 수 있다.
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu") # default가 cpu라서 꼭 gpu라고 줘야함
    parser.add_argument(
        "--data_dir",
        default="/data/ephemeral/data",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )
    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )
    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="best_model.pt", type=str, help="model file name"
    )
    parser.add_argument(
        "--output_dir", default="outputs/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    parser.add_argument(
        "--max_seq_len", default=20, type=int, help="max sequence length"
    )
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

    '''
    # [찬우] Data Augumentation 추가
    parser.add_argument(
        "--window", default=False, type=str2bool, help="window usage status"
    )
    parser.add_argument(
        "--stride", default=None, type=int, help="stride"
    )
    parser.add_argument(
        "--shuffle", default=False, type=str2bool, help="shuffle usage status"
    )
    parser.add_argument(
        "--shuffle_n", default=2, type=int, help="shuffle"
    )
    '''

    # [건우] feature 분류모음(자동화 위해 추가)
    '''
    parser.add_argument(
        "--cat_cols",
        #default=["testId", "assessmentItemID", "KnowledgeTag","paper_number"], # "userID"로 묶을 것이기 때문에 "userID"는 제외
        default=["assessmentItemID", "testId",
                 "KnowledgeTag"],
        type=list,
        help="categorical features",
    )
    parser.add_argument(
        "--con_cols",
        #default=["elapsed", "KnowledgeTag_percent", "cumulative"],
        default=["elapsed"],
        type=list,
        help="numerical features",
    )
    '''
    parser.add_argument(
        "--tgt_col", default=["answerCode"], type=list, help="target feature"
    )
    parser.add_argument(
        "--user_col", default=["userID"], type=list, help="target feature"
    )
    
    # 모델
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size"
    )
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.2, type=float, help="drop out rate")
    
    # Tfixup
    parser.add_argument("--Tfixup", default=False, type=bool, help="Tfirup layers")
    parser.add_argument("--Tfix_layer_norm", default=False, type=bool, help="Tfirup layers")
    parser.add_argument("--Tfix_n_layers", default=20, type=int, help="Tfirup layers")

    # 훈련
    parser.add_argument("--n_epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=10, type=int, help="clip grad")
    parser.add_argument("--patience", default=5, type=int, help="for early stopping")

    #### 옵션 ####
    # 공통
    parser.add_argument("--ratio", default=0.7, type=float, help="data split ratio")
    parser.add_argument("--shuffle", default=True, type=bool, help="data augmentation")
    # data augmentation
    parser.add_argument("--augmentation", default="", choices=['', 'window'], type=str, help="data augmentation")
    parser.add_argument("--stride", default=None, type=int, help="data augmentation")
    parser.add_argument("--shuffle_n", default=3, type=int, help="Mix data randomly and add them as data")
    # K-fold
    parser.add_argument("--kfold_splits", default=5, type=int, help="apply k-fold if 1 or more (minimum 5)")

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    ### 중요 ###
    parser.add_argument("--model", default="tfixup", choices=['lstm', 'lstmattn', 'bert', 'saint', 'lastquery, tfixup'],type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )

    # argparse.ArgumentParser()객체 안에 parse_args()라는 메소드가 있음 -> def parse_args()의 parse_args()가 아님
    args = parser.parse_args() # ex) Namespace(...,model='lstm', optimizer='adam', ...)

    return args # python script 실행할 때 인자를 수정하여 실행할 수 있다.
