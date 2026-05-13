from flask import Flask, render_template, jsonify, request
from datetime import datetime
from utils import (
    load_data,
    get_top_skills,
    jobs_over_time,
    salary_by_role,
    job_distribution,
    salary_distribution_box,
    salary_by_location,
    qualification_heatmap,
    job_distribution_by_work_type
)
from salary_model import SalaryModel

app = Flask(__name__)

# ====================================
# Load dataset + model ONLY ONCE
# ====================================

df = load_data()
model = SalaryModel(df)


# ====================================
# Filtering Helper
# ====================================

def get_filtered_df(base_df):

    filtered_df = base_df.copy()

    role = request.args.get('role', '').strip()
    location = request.args.get('location', '').strip()
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()

    if role:
        filtered_df = filtered_df[
            filtered_df['Job Title'] == role
        ]

    if location:
        filtered_df = filtered_df[
            filtered_df['location'] == location
        ]

    if start_date:

        start_dt = datetime.strptime(
            start_date,
            '%Y-%m-%d'
        )

        filtered_df = filtered_df[
            filtered_df['Job Posting Date'] >= start_dt
        ]

    if end_date:

        end_dt = datetime.strptime(
            end_date,
            '%Y-%m-%d'
        )

        filtered_df = filtered_df[
            filtered_df['Job Posting Date'] <= end_dt
        ]

    return filtered_df


# ====================================
# PAGES
# ====================================

@app.route('/')
def index():

    skills_data = get_top_skills(df)

    preview_items = list(
        skills_data.items()
    )[:5]

    top_skill = (
        preview_items[0][0]
        if preview_items else 'N/A'
    )

    stats = {

        'total_jobs': int(len(df)),

        'unique_roles': int(
            df['Job Title']
            .dropna()
            .nunique()
        ),

        'unique_locations': int(
            df['location']
            .dropna()
            .nunique()
        ),

        'top_skill': top_skill,

        'last_updated': datetime.now()
        .strftime('%d %b %Y')
    }

    preview_skills = {

        'labels': [
            item[0]
            for item in preview_items
        ],

        'values': [
            item[1]
            for item in preview_items
        ]
    }

    return render_template(
        'index.html',
        stats=stats,
        preview_skills=preview_skills
    )


@app.route('/dashboard')
def dashboard():

    roles = sorted(
        df['Job Title']
        .dropna()
        .unique()
        .tolist()
    )

    locations = sorted(
        df['location']
        .dropna()
        .unique()
        .tolist()
    )

    valid_dates = (
        df['Job Posting Date']
        .dropna()
    )

    min_date = (
        valid_dates.min().strftime('%Y-%m-%d')
        if not valid_dates.empty else ''
    )

    max_date = (
        valid_dates.max().strftime('%Y-%m-%d')
        if not valid_dates.empty else ''
    )

    return render_template(
        'dashboard.html',
        roles=roles,
        locations=locations,
        min_date=min_date,
        max_date=max_date
    )


@app.route('/salary-prediction')
@app.route('/prediction')
def salary_prediction():

    roles = (
        df['Job Title']
        .dropna()
        .unique()
    )

    locations = (
        df['location']
        .dropna()
        .unique()
    )

    return render_template(
        'prediction.html',
        roles=roles,
        locations=locations
    )


# ====================================
# API ROUTES
# ====================================

@app.route('/api/skills')
def skills():
    return jsonify(
        get_top_skills(
            get_filtered_df(df)
        )
    )


@app.route('/api/trends')
def trends():
    return jsonify(
        jobs_over_time(
            get_filtered_df(df)
        )
    )


@app.route('/api/salary')
def salary():
    return jsonify(
        salary_by_role(
            get_filtered_df(df)
        )
    )


@app.route('/api/job-distribution')
def job_distribution_data():
    return jsonify(
        job_distribution(
            get_filtered_df(df)
        )
    )


@app.route('/api/salary-distribution')
def salary_distribution():
    return jsonify(
        salary_distribution_box(
            get_filtered_df(df)
        )
    )


@app.route('/api/location-salary')
def location_salary():
    return jsonify(
        salary_by_location(
            get_filtered_df(df)
        )
    )


@app.route('/api/qualification-heatmap')
def qualification_heatmap_data():
    return jsonify(
        qualification_heatmap(
            get_filtered_df(df)
        )
    )


@app.route('/api/work-type-distribution')
def work_type_distribution():
    return jsonify(
        job_distribution_by_work_type(
            get_filtered_df(df)
        )
    )


@app.route('/api/jobs-by-country')
def jobs_by_country():

    filtered = get_filtered_df(df)

    if 'Country' not in filtered.columns:

        return jsonify({
            'countries': [],
            'values': []
        })

    temp = (
        filtered['Country']
        .dropna()
        .astype(str)
        .str.strip()
    )

    country_map = {

        "Venezuela, RB": "Venezuela",

        "Democratic Republic Of Congo":
        "Democratic Republic of the Congo",

        "British Virgin Islands":
        "Virgin Islands",

        "North Korea":
        "Korea, North",

        "South Korea":
        "Korea, South"
    }

    temp = temp.replace(country_map)

    data = temp.value_counts()

    return jsonify({

        'countries':
        data.index.tolist(),

        'values':
        data.values.tolist()
    })


# ====================================
# SALARY PREDICTION
# ====================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.json

        role = data.get('role')

        location = data.get('location')

        work_type = data.get(
            'work_type',
            ''
        )

        experience = float(
            data.get('experience', 0)
        )

        skills = data.get(
            'skills',
            ''
        )

        skills_list = [

            s.strip().lower()

            for s in skills.split(',')

            if s.strip()
        ]

        prediction = model.predict(

            role,
            location,
            work_type,
            experience,
            skills_list
        )

        if prediction is None:

            return jsonify({
                'error': 'Invalid input'
            }), 400

        return jsonify({

            'min':
            round(prediction * 0.9, 2),

            'max':
            round(prediction * 1.1, 2)
        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500


# ====================================
# TIMELINE PREDICTION
# ====================================

@app.route('/predict-timeline', methods=['POST'])
def predict_timeline():

    try:

        data = request.json

        role = data.get('role')

        location = data.get('location')

        work_type = data.get(
            'work_type',
            ''
        )

        skills = data.get(
            'skills',
            ''
        )

        skills_list = [

            s.strip().lower()

            for s in skills.split(',')

            if s.strip()
        ]

        # Reduced load
        experiences = list(
            range(0, 11, 2)
        )

        expected = []
        minimum = []
        maximum = []

        for exp in experiences:

            pred = model.predict(

                role,
                location,
                work_type,
                exp,
                skills_list
            )

            if pred is None:

                return jsonify({
                    'error': 'Invalid input'
                }), 400

            expected.append(
                round(pred, 2)
            )

            minimum.append(
                round(pred * 0.9, 2)
            )

            maximum.append(
                round(pred * 1.1, 2)
            )

        return jsonify({

            'experience':
            experiences,

            'expected':
            expected,

            'minimum':
            minimum,

            'maximum':
            maximum
        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500


# ====================================
# RUN APP
# ====================================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000
    )