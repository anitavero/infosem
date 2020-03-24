# Infosem

A toolkit to evaluate and visualise Embeddings.

## Requirements

* [wordcloud](https://amueller.github.io/word_cloud/)
* [unidecode](https://pypi.org/project/Unidecode/)

## Usage

Word cloud visualisation:

```
python visualise.py [-h] [--data-path DATA_PATH] [--save-name SAVE_NAME]
                    [-i INTERVAL] [-u URL_FILTER_PTRN] [--data-type DATA_TYPE]
                    [--action {wc_animation,month_freq_bar,word_hist,fb_msg_hist,embedding}]
                    [-l LANG] [--tn-dir TN_DIR]
                    [--tn-label {frequency,optics_cl}]
                    source

positional arguments:
  source                -

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        -
  --save-name SAVE_NAME
                        -
  -i INTERVAL, --interval INTERVAL
                        3000
  -u URL_FILTER_PTRN, --url-filter-ptrn URL_FILTER_PTRN
                        ''
  --data-type DATA_TYPE
                        'article'
  --action {wc_animation,month_freq_bar,word_hist,fb_msg_hist,embedding}
                        'wc_animation'
  -l LANG, --lang LANG  'english'
  --tn-dir TN_DIR       'tnboard_data'
  --tn-label {frequency,optics_cl}
                        'frequency'
```
