# sentiment_train.py
# Εκπαίδευση ταξινομητή συναισθήματος σε 3 κλίμακες (0=αρνητικό, 1=θετικό, 2=ουδέτερο)

# Εισάγουμε βιβλιοθήκες και εργαλεία για σε όλο το αρχείο
import pandas as pd
import numpy as np
# Εισάγουμε τη βιβλιοθήκη numpy που θα μας βοηθήσει αργότερα με τη γραμμική άλγεβρα

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

# DataSetChecker:έλεγχος sentences.csv, για στήλες text/label,τιμές (0,1,2)
import os
#βιβλιοθήκη για να μπορεί να ελέγχει αν ένα αρχείο υπάρχει
import pandas as pd

class DataSetChecker:
  #Δημιουργώ κλάση που ελέγχει κάθε πρόταση πριν την εκπαίδευση
  #το παρακάτω δεν αποτελεί σχόλιο αλλά κείμενο πειγραφής, που εξηγεί ακριβώς τι κάνει κάθε κλάση
    """
    Έλεγχοι:
    - Υπάρχει το αρχείο;
    - Διαβάζεται σωστά με UTF-8;
    - Υπάρχουν οι στήλες 'text' και 'label';
    - Υπάρχουν κενές τιμές σε αυτές τις στήλες;
    - Είναι οι ετικέτες μέσα στο φάσμα που εχω ορίει; (0,1,2);
    """

    def __init__(self,csv_path):
        self.csv_path = csv_path
      #με αυτόν τον τρόπο αποθηκεύω τη διαδρoμή του αρχείου που θα ελέγξω
    def file_exists(self)
      print("Βήμα 1: Έλεγχος ύπαρξης αρχεόυ.Αν το αρχείο υπάρχε επιστρέφει True ενώ σε δαφορετική περίπτωση False")
      return
      os.path.isfile(sewlf.csv_path)
    def load_csv(self):
        print("Βήμα 2: Φόρτωση του CSV σύμφωνα με την κωδκοποίηση UTF-8.")
      #Η pandas.read_csv φορτώνει το αρχεό σε ένα Data Frame
        return pd.read_csv(self.csv_path, encoding="utf-8")

    def check_columns(self, df):
      #ελέγχω αν υπάρχουν ο στήλες 'text','labels' που θέλω
       print("Βήμα 3: Έλεγχος σωστώ στηλών του π΄νακα)
            # Δίνω το σύνολο των στηλών που χρειάζονται
        required = {"text", "label"}
        #Υπολογίζω ποιες στήλες λείπουν από τον πίνακα
        missing = required - set(df.columns)
        #Δίνει το σύνολο με τς λέξεις που λείπουν σε διαφορετική περ΄πτωση που δεν λείπει καμία το σύνολο είνα κενό)
        return missing

    def check_missing_values(self, df):
    #Ελέγχει την ύπαρξη κενών τιμών
        print("Βήμα 4: Έλεγχος για κενές τιμέσ σε text και label.")
        missing_text = df["text"].isna().sum()
        missing_label = df["label"].isna().sum()
        return missing_text, missing_label

    
