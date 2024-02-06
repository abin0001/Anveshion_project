import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset
data = {
    'Patient_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    'Disease': ['Diabetes', 'Hypertension', 'Arthritis', 'Insomnia', 'Respiratory', 'Migraine', 'Digestive',
                'Skin Conditions', 'Stress', 'Weight Management', 'Allergies', 'Heart Health', 'Menstrual Health',
                'Liver Health', 'Immunity Boost', 'Cognitive Health', 'Joint Health', 'Respiratory Infections',
                'Hair Health', 'Eye Health', 'Kidney Health', 'Blood Pressure', 'Anxiety', 'Thyroid Health', 'Bone Health'],
    'Recommended_Herbs': [
        ['Thulasi', 'Neem', 'Amla'],
        ['Arjuna', 'Brahmi', 'Ashwagandha'],
        ['Turmeric', 'Ginger', 'Boswellia'],
        ['Valerian', 'Ashwagandha', 'Brahmi'],
        ['Tulsi', 'Licorice', 'Eucalyptus'],
        ['Peppermint', 'Lavender', 'Ginger'],
        ['Fennel', 'Ginger', 'Peppermint'],
        ['Aloe Vera', 'Neem', 'Turmeric'],
        ['Ashwagandha', 'Brahmi', 'Valerian'],
        ['Garcinia Cambogia', 'Triphala', 'Green Tea'],
        ['Turmeric', 'Ginger', 'Nettle'],
        ['Arjuna', 'Garlic', 'Hawthorn'],
        ['Shatavari', 'Ginger', 'Turmeric'],
        ['Milk Thistle', 'Dandelion', 'Turmeric'],
        ['Amla', 'Echinacea', 'Ashwagandha'],
        ['Brahmi', 'Ginkgo Biloba', 'Ashwagandha'],
        ['Turmeric', 'Ginger', 'Boswellia'],
        ['Tulsi', 'Ginger', 'Licorice'],
        ['Amla', 'Bhringraj', 'Neem'],
        ['Bilberry', 'Eyebright', 'Triphala'],
        ['Punarnava', 'Gokshura', 'Nettle'],
        ['Garlic', 'Hibiscus', 'Arjuna'],
        ['Ashwagandha', 'Lavender', 'Passionflower'],
        ['Ashwagandha', 'Bladderwrack', 'Turmeric'],
        ['Calcium-rich foods', 'Turmeric', 'Ginger']
    ]
}

df = pd.DataFrame(data)

df.to_csv('recomend_data.csv')

# # Combine the Recommended_Herbs list into a single string for vectorization
# df['Herbs_Text'] = df['Recommended_Herbs'].apply(lambda herbs: ' '.join(herbs))

# # Use CountVectorizer to convert text data into a matrix of token counts
# vectorizer = CountVectorizer()
# herbs_matrix = vectorizer.fit_transform(df['Herbs_Text'])

# # Calculate cosine similarity between herb vectors
# cosine_sim = cosine_similarity(herbs_matrix, herbs_matrix)

# def recommend_herbs(patient_id):
#     # Get the index of the patient
#     index = df[df['Patient_ID'] == patient_id].index[0]

#     # Get the cosine similarity scores for the patient's herbs
#     sim_scores = list(enumerate(cosine_sim[index]))

#     # Sort the herbs based on similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Get the indices of the top 3 similar patients (excluding itself)
#     top_indices = [i[0] for i in sim_scores[1:4]]

#     # Return the recommended herbs for the top similar patients
#     return df.iloc[top_indices]['Recommended_Herbs'].tolist()

# # Example usage
# patient_id = 1
# recommendations = recommend_herbs(patient_id)
# print(f"Recommended Herbs for Patient {patient_id}: {recommendations}")
