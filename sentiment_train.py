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
# Προσπαθουμε να διαβάσουμε  το sentences.csv σε έναν πίνακα, σε περίπτωση που το αρχείο λείπει θα εμφανιστεί κάποιο σφάλμα
data = pd.read_csv("sentences.csv") 

# DataSetChecker:έλεγχος sentences.csv, για στήλες text/label,τιμές (0,1,2)
import os
#με την εντολή αυτή φέρνουμε τη βιβλιοθήκη os που μας δίνει πρόσβαση σε αρχεία
import pandas as pd

class DataSetChecker:
  #Δημιουργώ κλάση που έχει σαν σκοπό τον έλεγχο κάθε πρότασης πριν την εκπαίδευση
  #το παρακάτω δεν αποτελεί σχόλιο αλλά κείμενο πειγραφής, που εξηγεί ακριβώς τι κάνει κάθε κλάση
    """
    Έλεγχοι:
    - Υπάρχει το αρχείο;
    - Διαβάζεται σωστά με UTF-8;
    - Υπάρχουν οι στήλες 'text' και 'label';
    - Υπάρχουν κενές τιμές σε αυτές τις στήλες;
    - Είναι οι ετικέτες μέσα στο φάσμα που εχω ορίει; (0,1,2);
    """
    def __init__(self, csv_path):
        #Αποθηκεύω την διαδρομή του αρχείου CSV που θα ελέγξω.
        self.csv_path = csv_path

    def file_exists(self):
        # Βήμα 1: Έλεγχος αν υπάρχει το αρχείο 
        print("Βήμα 1: Έλεγχος ύπαρξης αρχείου.")
        # Η os.path.isfile επιστρέφει True αν το αρχείο υπάρχει σε διαφορετική περίπτωση επιστρέφει False.
        return os.path.isfile(self.csv_path)

    def load_csv(self):
      #Η μέθοδος def ανοίγει και είναι σε θέση να διαβάσει CSV
      # Βήμα 2: Προσπάθεια φόρτωσης του αρχείου CSV με κωδικοποίηση UTF-8, αυτό γιατί έχουμε ελληνικά και είναι πολύ σημαντική η σωστή κωδικοποίηση
        print("Βήμα 2: Φόρτωση του CSV με κωδικοποίηση UTF-8.")
        # Η pandas διαβάζει το αρχείο και το κάνει πίνακα(DataFrame)
        return pd.read_csv(self.csv_path, encoding="utf-8")

    def check_columns(self, df):
        # Βήμα 3: Έλεγχος αν υπάρχουν οι απαραίτητες στήλες 'text' και 'label'.
        print("Βήμα 3: Έλεγχος σωστών στηλών του πίνακα.")
        required = {"text", "label"}
        #Φτιάχνω set με τα ονόματα των στηλών που θεωρώ υποχρεωτικές
         missing = required - set(df.columns)
      # Υπολογίζω ποιες από τις απαιτούμενες(required)στήλες λείπουν από τον πίνακα.
        # Επιστρέφω το σύνολο με τις στήλες που λείπουν (αν δεν λείπει καμία, το σύνολο είναι κενό)()
        return missing

    def check_missing_values(self, df):
        # Βήμα 4: Έλεγχος για κενές τιμές στις στήλες 'text' και 'label'.
        print("Βήμα 4: Έλεγχος για κενά πεδία σε text και label.")
        # Υπολογίζω πόσες κενές τιμές υπάρχουν στη στήλη 'text'.
        missing_text = df["text"].isna().sum()
        #παίρνω τη στήλη text
        # Υπολογίζω πόσες κενές τιμές υπάρχουν στη στήλη 'label'.
        missing_label = df["label"].isna().sum()
        #η .insa φτιάχνει μία στήλη True/False, και .sum η μετράει πόσα true 
        # Επιστρέφω τον αριθμό κενών τιμών για κάθε στήλη.
        return missing_text, missing_label

    def check_label_values(self, df):
        # Βήμα 5: Έλεγχος για άκυρες ετικέτες στη στήλη 'label'.
        print("Βήμα 5: Έλεγχος ετικετών στη στήλη label.")
        valid_labels = {0, 1, 2}
        # επιτρέπει τιμές γαι την ετικέτα συναιθήματος .
        found_labels = set(df["label"].unique())
        # Υπολογίζω ποιες από τις ετικέτες που έχουν βρεθεί δεν ανήκουν στο σύνολο των επιτρεπτών.
        invalid = found_labels - valid_labels
        # Επιστρέφω το σύνολο με τις άκυρες ετικέτες. Οι άκυρες ετικέτες είναι όποιες δεν έιναι 0/1/2 Σε περίπτωση που είναι όλες σωστές μου επιστρέφει κενό
        return invalid

    def run_all_checks(self):
        # Αυτή η συνάρτηση τρέχει όλους τους ελέγχους με τη σειρά
        print("Ξεκινάει ο εξ ολοκλήρου έλεγχος του αρχείου sentences.csv.")

        # Βήμα 1: Έλεγχος αν το αρχείο υπάρχει.
        if not self.file_exists():
            print("ΛΑΘΟΣ: Το αρχείο δεν βρέθηκε")
            return False

        # Βήμα 2: Φόρτωση του CSV σε DataFrame.
        df = self.load_csv()

        # Βήμα 3: Έλεγχος αν λείπουν στήλες.
        missing_columns = self.check_columns(df)
        if missing_columns:
            print("ΛΑΘΟΣ: Λείπουν οι παρακάτω στήλες από το dataset:", missing_columns)
            return False
          #Αν δεν είναι κενό τυπώνει ποιες λείπουν

        # Βήμα 4: Έλεγχος για κενές τιμές στις στήλες 'text' και 'label'.
        missing_text, missing_label = self.check_missing_values(df)
        if missing_text > 0 or missing_label > 0:
          #Ουσιαστικά παίρνω 2 αριθμούς αν είανι μικρότερος απο 0 βλέπω πόσα κενά υπαρχουν
            print(f"ΛΑΘΟΣ: Κενές τιμές στη στήλη text: {missing_text}, "
                  f"κενές τιμές στη στήλη label: {missing_label}.")
            return False
 
         def run_all_checkers(self):
          #εκτελεί όλους τους ελέγχους για το αρχείο csv
           print("Τώρα αρχίζει ο συνολικός έλεγχος του αρχείου csv")
           def run_all_checkers(self):
        # Συνάρτηση που εκτελεί ΟΛΟΥΣ τους ελέγχους για το αρχείο sentences.csv.
        # Χρησιμοποιώ μια μεταβλητή all_good για να θυμάμαι
        # αν μέχρι τώρα όλα είναι εντάξει (True) ή αν βρέθηκε κάποιο λάθος (False).

        print("=== Ξεκινάει ο συνολικός έλεγχος του αρχείου sentences.csv ===")

        # Στην αρχή προυποθέτω ότι όλα είναι εντάξει με τη μεταβλητή boolean (all_good)
        all_good = True
        #καλώ την μέθοδο file_exists που έχω ορίσει παραπάνω
      # Μέρος 1
        if not self.file_exists():
            # Αν ΔΕΝ υπάρχει το αρχείο,τότε τυπώνω 
            print("ΛΑΘΟΣ: Το αρχείο sentences.csv δεν βρέθηκε στον φάκελο.")
            # αυτή η μέθοδος γυρνάει True αν υπάρχει το αρχείο και False αν δεν υπάρχει
            all_good = False
        else:
            print("OK: Το αρχείο sentences.csv βρέθηκε.")
        # Μέρος 2: Φόρτωση του CSV σε DataFrame, μόνο σε περίπτωση που όλα μέχρι τώρα είναι καλά
        #Φτιάχνω μία μεταβλητή df και της δίνεις τιμή "άδειο" δηλ.None
        #Αν η φόρτωση πετύχει η df θα γίνει ένας πίνακας (Data Frame)
        df = None 
       #Αν μέχρι στιγμής όλα καλά δηλαδή ακόμα True προχωράμε στη φόρτωση
        if all_good:
            df = self.load_csv()
            print("OK: Το αρχείο φορτώθηκε σωστά σε πίνακα")
        def load_csv(self):
        #Ορίζει μέθοδο μέσα στην κλάση DataSetChecker
        # Φορτώνει το CSV σε DataFrame με κωδικοποίηση UTF-8.
        print("Φόρτωση CSV από το μονοπάτι:" + self.csv_path)
        df = pd.read_csv(self.csv_path, encoding="utf-8")
       #επιστρέφει τον πίνακα df
      return df

