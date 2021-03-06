# Infosem

A toolkit to evaluate and visualise Embeddings.

## Requirements

* [wordcloud](https://amueller.github.io/word_cloud/)
* unidecode
* matplotlib
* argh
* numpy
* tqdm

To install the required packages run `pip install requirements.txt`

## Usage

### Word cloud visualisation

This functionality generates a video of wordclouds of Facebook or Slack messages. 
Each video frame will show a wordcloud for a given day.

The messages can be dowloaded from the respective websites.

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
