def review_summary(sentence_list):

#  from transformers import pipeline

  classifier = pipeline('sentiment-analysis')
  results = classifier(sentence_list)

  sent_pos = []
  sent_neg = []

  proba_pos = []
  proba_neg = []

  for result in results:
    if(result['label'] == 'POSITIVE'):
      sent_pos.append(result)
      proba_pos.append(result['score'])
    else:
      sent_neg.append(result)
      proba_neg.append(result['score'])
  
  sentiment_pos = sum(proba_pos) / len(sentence_list)
  sentiment_neg = sum(proba_neg) / len(sentence_list)

  probability = 0
  sentiment = ''

  if (sentiment_pos > sentiment_neg):
    probability = sentiment_pos
    sentiment = 'positive'
  else: 
    probability = sentiment_neg
    sentiment = 'negative'

  adj = ""

  if (probability < 0.25):
    adj = 'slightly'
  elif (probability >= 0.25 and probability < 0.5):
    adj = 'fairly'
  elif (probability >= 0.5 and probability < 0.75):
    adj = 'considerably'
  elif (probability > 0.75):
    adj = 'highly'

  print(f'positive count / probabiltiy: {len(sent_pos)} sentence(s), {round(sentiment_pos, 3) * 100} %')
  print(f'negative count / probabiltiy: {len(sent_neg)} sentence(s), {round(sentiment_neg, 3) * 100} %')
  print(f'This review is {adj} {sentiment}.')
 



''' 
EXAMPLE summary. (raw text from rotton tomato critiques)

positive count / probabiltiy: 36 sentence(s), 67.5 %
negative count / probabiltiy: 15 sentence(s), 28.7 %
This review is considerably positive.

'''
