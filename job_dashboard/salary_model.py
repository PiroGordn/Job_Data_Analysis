from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pandas as pd
import numpy as np
import re


class SalaryModel:

    def __init__(self, df):

        print("Initializing SalaryModel...")

        # Keep required columns only
        self.df = df[
            ['Job Title', 'location', 'avg_salary', 'skills']
        ].dropna().copy()

        # Optional sampling for faster training
        if len(self.df) > 5000:
            self.df = self.df.sample(
                n=5000,
                random_state=42
            )

        # Encoders
        self.le_role = LabelEncoder()
        self.le_location = LabelEncoder()

        # Encode categorical columns
        self.df['role_enc'] = self.le_role.fit_transform(
            self.df['Job Title'].astype(str)
        )

        self.df['loc_enc'] = self.le_location.fit_transform(
            self.df['location'].astype(str)
        )

        # Features and target
        X = self.df[['role_enc', 'loc_enc']]
        y = self.df['avg_salary']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Train model
        self.model = LinearRegression()

        self.model.fit(X_train, y_train)

        # Predictions
        y_pred = self.model.predict(X_test)

        self.skill_salary_map = {}

        temp = self.df[['skills', 'avg_salary']].dropna()

        skill_totals = {}
        skill_counts = {}

        for _, row in temp.iterrows():

            # Supports commas, pipes, semicolons, etc.
            skills = re.split(
                r'[,|;/]',
                str(row['skills'])
            )

            for skill in skills:

                skill = skill.strip().lower()

                if not skill:
                    continue

                skill_totals[skill] = (
                    skill_totals.get(skill, 0)
                    + row['avg_salary']
                )

                skill_counts[skill] = (
                    skill_counts.get(skill, 0)
                    + 1
                )

        for skill in skill_totals:

            self.skill_salary_map[skill] = (
                skill_totals[skill]
                / skill_counts[skill]
            )

        print(
            "TOTAL SKILLS:",
            len(self.skill_salary_map)
        )

        print("SalaryModel ready")

    def predict(
        self,
        role,
        location,
        work_type=None,
        experience=0,
        skills=None
    ):

        role = str(role).strip()
        location = str(location).strip()

        # Validate inputs
        if role not in self.le_role.classes_:
            return None

        if location not in self.le_location.classes_:
            return None

        # Encode
        role_enc = self.le_role.transform([role])[0]
        loc_enc = self.le_location.transform([location])[0]

        # Base prediction
        base = float(
            self.model.predict(
                [[role_enc, loc_enc]]
            )[0]
        )
        experience = max(
            0,
            float(experience)
        )

        base *= (1 + 0.03 * experience)

        if work_type == "Part-Time":
            base *= 0.7

        elif work_type == "Contract":
            base *= 1.1

        elif work_type == "Intern":
            base *= 0.5

        if skills:

            bonuses = []

            for skill in skills:

                skill = skill.lower().strip()

                print("CHECKING:", skill)

                if skill in self.skill_salary_map:

                    print("MATCHED:", skill)

                    bonuses.append(
                        self.skill_salary_map[skill]
                    )

            if bonuses:

                avg_skill_salary = (
                    sum(bonuses)
                    / len(bonuses)
                )

                adjustment = (
                    avg_skill_salary / base
                )

                # Prevent excessive changes
                adjustment = min(
                    max(adjustment, 0.9),
                    1.3
                )

                print(
                    "SKILL ADJUSTMENT:",
                    adjustment
                )

                base *= adjustment

        return base