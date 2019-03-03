import os

dir_logs = r'C:\development\crodnet\experiments\2019'

text_to_search = 'Val accuracy_conf: 0.91'

for elem in os.listdir(dir_logs):
    if os.path.isdir(os.path.join(dir_logs, elem)):
        log_file = os.path.join(dir_logs, elem, 'out.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as fid:
                lines = fid.read().split('\n')
            for line in lines:
                if text_to_search in line:
                    print('Text found in file ' + log_file)
                    break
