from engine import LasEngine
import argparse 
import joblib

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str)

paras = parser.parse_args()

cp_path = '/home/datvt/End-to-end-ASR-Pytorch/result/libri_example_sd0/asr'
mapper_path = '/home/datvt/data'

engine = LasEngine.load(model_path=cp_path, mapper_path=mapper_path)

valid_path = '/home/common/corpora/speech/valid_corpus/vov_valid_corpus.txt'

content = open(valid_path, 'r').readlines()
f = open(valid_path.split('/')[-1][:-11] + '.txt', 'w+')

#def transc(engi, fout, wav_path, transcript):
#    print(transcript)
#    ans_len = len(transcript) + 50
#    pred = engi.transcribe(wav_path, ans_len)
#    f.write(f'{pred}|{transcript}\n')

#wav_paths = [x.split('|')[0] for x in content]
#transcripts = [x.split('|')[-1].replace('\n', '') for x in content]

#wav_paths = wav_paths[:50]
#transcripts = transcripts[:50]

#joblib.Parallel(n_jobs = 4)(joblib.delayed(transc)(engine, f, wp, tr) for wp, tr in zip(wav_paths,transcripts))

for i, v in enumerate(content):
    if i % 100 == 0:
        print(i)
    wav_path, transcript = v.split('|')
    transcript = transcript.replace('\n', '')
    ans_len = len(transcript) + 50
    pred = engine.transcribe(wav_path, ans_len)
    f.write(f'{pred}|{transcript}\n')
#    if i > 50:
#        break

