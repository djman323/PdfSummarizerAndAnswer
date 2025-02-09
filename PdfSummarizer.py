import fitz  # PyMuPDF for PDF text extraction
import numpy as np
import nltk
import re
import networkx as nx
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self.extract_text()
        self.sentences = sent_tokenize(self.text)

    def extract_text(self):
        """Extract text from the given PDF file."""
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()

class TextSummarizer:
    def __init__(self, sentences):
        self.sentences = sentences
        self.summary = self.summarize()

    def sentence_similarity(self, sent1, sent2):
        """Compute cosine similarity between two sentences."""
        stop_words = set(stopwords.words("english"))
        sent1 = [word for word in word_tokenize(sent1) if word.lower() not in stop_words]
        sent2 = [word for word in word_tokenize(sent2) if word.lower() not in stop_words]
        all_words = list(set(sent1 + sent2))
        
        vector1 = [1 if word in sent1 else 0 for word in all_words]
        vector2 = [1 if word in sent2 else 0 for word in all_words]
        
        return cosine_similarity([vector1], [vector2])[0][0]

    def build_similarity_matrix(self):
        """Create a similarity matrix for all sentences."""
        num_sentences = len(self.sentences)
        similarity_matrix = np.zeros((num_sentences, num_sentences))

        for i in range(num_sentences):
            for j in range(num_sentences):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity(self.sentences[i], self.sentences[j])
        
        return similarity_matrix

    def summarize(self, top_n=3):
        """Summarize the text using TextRank."""
        if not self.sentences:
            return "No text available for summarization."

        similarity_matrix = self.build_similarity_matrix()
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)

        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(self.sentences)), reverse=True)
        summary = " ".join([ranked_sentences[i][1] for i in range(min(top_n, len(ranked_sentences)))])
        return summary

class QuestionAnswering:
    def __init__(self, sentences):
        self.sentences = sentences
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.sentences)

    def answer_question(self, question):
        """Find the best answer from the text using TF-IDF cosine similarity."""
        if not self.sentences:
            return "No content available to answer questions."

        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.tfidf_matrix).flatten()
        best_match_index = np.argmax(similarities)

        return self.sentences[best_match_index]

# Tkinter GUI
class PDFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Summarizer & QnA")
        self.root.geometry("700x500")

        self.pdf_path = ""
        self.sentences = []

        # Buttons
        self.upload_btn = tk.Button(root, text="Upload PDF", command=self.upload_pdf, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.upload_btn.pack(pady=10)

        self.summarize_btn = tk.Button(root, text="Generate Summary", command=self.generate_summary, font=("Arial", 12), bg="#2196F3", fg="white")
        self.summarize_btn.pack(pady=5)

        # Text Box for Output
        self.text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=15, font=("Arial", 11))
        self.text_box.pack(pady=10)

        # Question Answering Section
        self.question_label = tk.Label(root, text="Ask a question:", font=("Arial", 12))
        self.question_label.pack()

        self.question_entry = tk.Entry(root, width=50, font=("Arial", 12))
        self.question_entry.pack(pady=5)

        self.answer_btn = tk.Button(root, text="Get Answer", command=self.answer_question, font=("Arial", 12), bg="#FF5722", fg="white")
        self.answer_btn.pack(pady=5)

    def upload_pdf(self):
        """Open file dialog to select PDF and extract text."""
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.pdf_path = file_path
            processor = PDFProcessor(self.pdf_path)
            self.sentences = processor.sentences
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, processor.text[:1000] + "\n\n...(Text truncated for display)...")
            messagebox.showinfo("Success", "PDF uploaded successfully!")

    def generate_summary(self):
        """Generate a summary of the extracted text."""
        if not self.sentences:
            messagebox.showwarning("Warning", "No PDF uploaded!")
            return

        summarizer = TextSummarizer(self.sentences)
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, "### Summary ###\n" + summarizer.summary)

    def answer_question(self):
        """Provide an answer based on the extracted text."""
        if not self.sentences:
            messagebox.showwarning("Warning", "No PDF uploaded!")
            return

        question = self.question_entry.get()
        if not question.strip():
            messagebox.showwarning("Warning", "Please enter a question!")
            return

        qa_system = QuestionAnswering(self.sentences)
        answer = qa_system.answer_question(question)
        messagebox.showinfo("Answer", answer)

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = PDFApp(root)
    root.mainloop()
