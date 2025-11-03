03-11-2025
Βιβλιοθήκες για Project(κατά σειρά). με τη σειρά που θα χρησιμοποιηθούν

re → Βιβλιοθήκη προκειμένου να αφαιρεθούν σύμβολα, emojis, σημεία στίξης. κανονικών εκφράσεων (regular expressions)
import re

nltk → Natural Language Toolkit.Βιβλιοθήκη προκειμένου να κάνουμε tokenization και να αφαιρέσουμε τις 'stopwords',
που δεν μας παρέχουν σημασιολογικό βάρος.
import nltk

Pandas → Βιβλιοθήκη που μας βοηθά με τη διαχείριση δεδομένων είτε αυτά είναι σε μορφή πίνακα είτε όχι.
import pandas as pd

numpy → Παρέχει βοήθεια στον υπολογισμό πινάκων
import numpy as np

scikit-learn →μας βοηθά στο στάδιο ταξινόμησης συναισθήματος
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


tensorflow → Για πρόβλεψη συναισθήματος
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
