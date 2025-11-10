# sentiment_train.py
# Εκπαίδευση ταξινομητή συναισθήματος σε 3 κλίμακες (0=αρνητικό, 1=θετικό, 2=ουδέτερο)

# Εισάγουμε βιβλιοθήκες και εργαλεία για σε όλο το αρχείο
import pandas as pd

# από το scikit-learn: κάνει μετατροπή κειμένου σε διανύσματα 
from sklearn.model_selection import train_test_split            
from sklearn.feature_extraction.text import TfidfVectorizer     
from sklearn.linear_model import LogisticRegression             
from sklearn.naive_bayes import MultinomialNB                   
from sklearn.metrics import classification_report, confusion_matrix 

from text_prepare import clean_text


# στη συνέχεια φορτώνουμε τα δεδομένα μας
# Προσπαθουμε να διαβάσουμε  το sentences.csv σε έναν πίνακα, σε περίπτωση που το αρχείο λείπει θα εμφανιστεί κα΄ποιο σφάλμα
data = pd.read_csv("sentences.csv") 
# πρέπει να βρω κάποιον τρόπο να γίνει έλεγχος ορθότητας στηλών και και ετικετών!






