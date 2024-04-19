from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
import math

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/alumni_db'
db = SQLAlchemy(app)

# Define the Careers model
class Careers(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_title = db.Column(db.String(255))
    skills = db.Column(db.String(255))
    company = db.Column(db.String(255))
    location = db.Column(db.String(255))
    description = db.Column(db.Text)

@app.route('/')
def work():
    return render_template('work.html')

def calculate_cosine_similarity(job_features, user_skills_str):
    # Calculate TF-IDF matrix for job features
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_features)

    # Get the TF-IDF vectors
    user_tfidf_vector = tfidf_matrix[-1].toarray()[0]
    job_tfidf_vectors = tfidf_matrix[:-1].toarray()

    # Calculate cosine similarity manually
    similarity_scores = []
    for job_tfidf_vector in job_tfidf_vectors:
        dot_product = sum(a * b for a, b in zip(user_tfidf_vector, job_tfidf_vector))
        user_magnitude = math.sqrt(sum(a ** 2 for a in user_tfidf_vector))
        job_magnitude = math.sqrt(sum(b ** 2 for b in job_tfidf_vector))
        similarity = dot_product / (user_magnitude * job_magnitude)
        similarity_scores.append(similarity)

    return similarity_scores

@app.route('/search', methods=['GET'])
def search_jobs():
    try:
        user_skills = request.args.get('skills')
        if not user_skills:
            return jsonify({'error': 'Please provide skills in the query parameters.'}), 400

        user_skills_list = [skill.strip().lower() for skill in user_skills.split(',')]

        # Get all jobs from the database
        all_jobs = Careers.query.all()
        job_data = [(job.job_title, job.description, job.skills, job.company, job.location) for job in all_jobs]

        # Prepare data for TF-IDF vectorization
        job_features = [' '.join(job) for job in job_data]
        user_skills_str = ' '.join(user_skills_list)
        job_features.append(user_skills_str)

        # Calculate cosine similarity
        similarity_scores = calculate_cosine_similarity(job_features, user_skills_str)

        # Combine similarities with job data
        job_similarities = list(zip(job_data, similarity_scores))
        job_similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top 5 recommended jobs
        recommended_jobs = [
            {'job_title': job[0], 'job_description': job[1], 'required_skills': job[2], 'company': job[3], 'location': job[4], 'similarity_score': score}
            for (job, score) in job_similarities[:5]
        ]

        return jsonify({'recommended_jobs': recommended_jobs})
    except Exception as e:
        print(f'Error processing job search: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)