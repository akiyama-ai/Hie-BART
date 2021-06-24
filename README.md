# Hie-BART
As a preprocessing step, you need to give a CLS token at the beginning of each sentence in the source document using **hie_make_datafiles.py**.

# Scripts
pytorch=1.4.0

Follow the instructions of fairseq=0.9.0 (https://github.com/pytorch/fairseq/blob/v0.9.0/examples/bart/README.cnn.md)

## Downloads
- bart.large (https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz)
- CNN_DailyMail Datasets

## Commands
1. command_hie_bpe.sh       : Apply BPE to the data set

2. command_hie_binarize.sh  : Binarize the data set

3. command_fine-tuning.sh   : Fine-tuning

4. inference_hie.py         : Inference

5. command_hie_rouge.sh     : Calculate ROUGE score
  
