# Infosem

A toolkit to evaluate and visualise Embeddings.

## Requirements

* [wordcloud](https://amueller.github.io/word_cloud/)
* unidecode
* matplotlib
* argh
* numpy
* tqdm

## Usage

Word cloud visualisation:

```
python visualise.py [-h] [--source {fb,slack}] [--data_path DATA_PATH]
                    [--save_name SAVE_NAME] [--interval INTERVAL]
                    [--url_filter_ptrn URL_FILTER_PTRN]
                    [--lang LANG] [--tn-dir TN_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --source {fb,slack}   Message source. "fb": Facebook or "slack" (default:
                        'fb')
  --data_path DATA_PATH
                        Full path to the data directory.
  --save_name SAVE_NAME
                        Full path to the video file we save.
  --interval INTERVAL   Interval between video frames in miliseconds.
                        (default: 3000)
  --url_filter_ptrn URL_FILTER_PTRN
                        Pattern to filter urls. (default: '')
  --lang LANG           Language of the messages. "english", "hungarian" or
                        "hunglish". (default: 'english')
```