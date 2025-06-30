#!/bin/bash

# Directory paths
DATA_DIR="../data"
MODEL_DIR="./models"
EVAL_DIR="./evals"
LOG_DIR="./logs"

mkdir -p $MODEL_DIR
mkdir -p $EVAL_DIR
mkdir -p $LOG_DIR

# Experiment configurations
TRAIN_DATA="$DATA_DIR/ensup-tiny"   # Using ensup-tiny as in your example
DEV_DATA="$DATA_DIR/endev"

# Hyperparameters to experiment with
RNN_DIMS=(5 10 20)
LEARNING_RATES=(0.001 0.0005)
REGULARIZATIONS=(0.0001 0.00005)
BATCH_SIZES=(1 16)

# Suppress warnings and progress bars
export PYTHONWARNINGS="ignore"
export DISABLE_TQDM=1

# -------------------------------
# biRNN-CRF experiments (Neural Models)
# -------------------------------
for RNN_DIM in "${RNN_DIMS[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        for REG in "${REGULARIZATIONS[@]}"; do
            for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                MODEL_NAME="en_crf_neural_rnn${RNN_DIM}_lr${LR}_reg${REG}_bs${BATCH_SIZE}.pkl"
                EVAL_FILE="$EVAL_DIR/en_crf_neural_rnn${RNN_DIM}_lr${LR}_reg${REG}_bs${BATCH_SIZE}.eval"
                LOG_FILE="$LOG_DIR/en_crf_neural_rnn${RNN_DIM}_lr${LR}_reg${REG}_bs${BATCH_SIZE}.log"

                echo "Running biRNN-CRF with rnn_dim=${RNN_DIM}, lr=${LR}, reg=${REG}, batch_size=${BATCH_SIZE}"
                python3 tag.py $DEV_DATA \
                    --train $TRAIN_DATA \
                    --model $MODEL_DIR/$MODEL_NAME \
                    --crf \
                    --rnn_dim $RNN_DIM \
                    --eval_file $EVAL_FILE \
                    --eval_interval 500 \
                    --lr $LR \
                    --reg $REG \
                    --batch_size $BATCH_SIZE \
                    --device cpu \
                    --max_steps 5000 \
                    > $LOG_FILE 2>&1

                # Extract cross-entropy and tagging accuracy from EVAL_FILE
                CROSS_ENTROPY_LINE=$(grep "Cross-entropy" $EVAL_FILE | tail -1)
                ACCURACY_LINE=$(grep "Tagging accuracy" $EVAL_FILE | tail -1)
                echo "Results for biRNN-CRF with rnn_dim=${RNN_DIM}, lr=${LR}, reg=${REG}, batch_size=${BATCH_SIZE}:"
                echo "$CROSS_ENTROPY_LINE"
                echo "$ACCURACY_LINE"
                echo "---------------------------------------------"
            done
        done
    done
done

# -------------------------------
# Simple CRF experiments
# -------------------------------
for LR in "${LEARNING_RATES[@]}"; do
    for REG in "${REGULARIZATIONS[@]}"; do
        MODEL_NAME="en_crf_backprop_lr${LR}_reg${REG}.pkl"
        EVAL_FILE="$EVAL_DIR/en_crf_backprop_lr${LR}_reg${REG}.eval"
        LOG_FILE="$LOG_DIR/en_crf_backprop_lr${LR}_reg${REG}.log"

        echo "Running Simple CRF with lr=${LR}, reg=${REG}"
        python3 tag.py $DEV_DATA \
            --train $TRAIN_DATA \
            --model $MODEL_DIR/$MODEL_NAME \
            --crf \
            --eval_file $EVAL_FILE \
            --eval_interval 500 \
            --lr $LR \
            --reg $REG \
            --device cpu \
            --max_steps 5000 \
            > $LOG_FILE 2>&1

        # Extract cross-entropy and tagging accuracy from EVAL_FILE
        CROSS_ENTROPY_LINE=$(grep "Cross-entropy" $EVAL_FILE | tail -1)
        ACCURACY_LINE=$(grep "Tagging accuracy" $EVAL_FILE | tail -1)
        echo "Results for Simple CRF with lr=${LR}, reg=${REG}:"
        echo "$CROSS_ENTROPY_LINE"
        echo "$ACCURACY_LINE"
        echo "---------------------------------------------"
    done
done

