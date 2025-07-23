from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Reference and generated texts
reference = " An adult panda can grow up to 4 to 6 feet long and weigh up to 250 pounds. "
generated = """ Pandas are large mammals with a weight range of about 200-450 pounds (90-200 kg). The average weight for a female
pandas ranges between 160 and 370 pounds (72.8 and 168 kg), while males can weigh up to 200 pounds (90 kg) or
more. They are also very big, standing at about 4 feet (123 cm) tall at the shoulder and weighing between 15-20
pounds (7.2-9.1 kg). The life span of a pandas is up to 18 years in the wild, but they typically live for several
decades or longer in captivity. """

# BLEU Score with smoothing
smoothie = SmoothingFunction().method4
bleu = sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothie)
print("BLEU Score:", round(bleu, 4))

# ROUGE Scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, generated)
print("ROUGE Scores:", scores)
