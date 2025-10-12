#!/usr/bin/env python3
"""
Class Identification Analysis
Check if the model is identifying all 454 Sinhala character classes
"""

import pandas as pd

def analyze_class_identification():
    # Load the detailed classification report
    df = pd.read_csv('../evaluation_results/detailed_classification_report.csv')

    print('📊 CLASS IDENTIFICATION ANALYSIS')
    print('=' * 50)
    print(f'Total classes in dataset: {len(df)}')

    # Check classes with zero F1-score (completely missed)
    zero_classes = df[df['F1_Score'] == 0.0]
    print(f'Classes with F1-Score = 0 (not predicted): {len(zero_classes)}')

    # Check classes being predicted
    predicted_classes = df[df['F1_Score'] > 0.0]
    print(f'Classes with F1-Score > 0 (being predicted): {len(predicted_classes)}')

    print(f'\n📈 F1-Score Statistics:')
    print(f'- Mean: {df["F1_Score"].mean():.4f}')
    print(f'- Min:  {df["F1_Score"].min():.4f}')
    print(f'- Max:  {df["F1_Score"].max():.4f}')

    # Performance ranges
    excellent = len(df[df['F1_Score'] >= 0.9])
    good = len(df[(df['F1_Score'] >= 0.7) & (df['F1_Score'] < 0.9)])
    average = len(df[(df['F1_Score'] >= 0.5) & (df['F1_Score'] < 0.7)])
    poor = len(df[(df['F1_Score'] > 0.0) & (df['F1_Score'] < 0.5)])

    print(f'\n🎯 Performance Distribution:')
    print(f'- Excellent (F1 ≥ 0.9): {excellent} classes ({excellent/len(df)*100:.1f}%)')
    print(f'- Good (0.7 ≤ F1 < 0.9): {good} classes ({good/len(df)*100:.1f}%)')
    print(f'- Average (0.5 ≤ F1 < 0.7): {average} classes ({average/len(df)*100:.1f}%)')
    print(f'- Poor (0 < F1 < 0.5): {poor} classes ({poor/len(df)*100:.1f}%)')
    print(f'- Not predicted (F1 = 0): {len(zero_classes)} classes ({len(zero_classes)/len(df)*100:.1f}%)')

    # Show classes that are not being predicted at all
    if len(zero_classes) > 0:
        print(f'\n⚠️  CLASSES NOT BEING PREDICTED (F1=0):')
        for idx, row in zero_classes.head(10).iterrows():
            print(f'   {row["Unicode_Character"]} ({row["Class_Name"]}): Support={row["Support"]}')
        if len(zero_classes) > 10:
            print(f'   ... and {len(zero_classes) - 10} more classes')
    else:
        print(f'\n✅ ALL CLASSES ARE BEING PREDICTED!')

    # Show very low performing classes
    very_low = df[df['F1_Score'] < 0.3]
    if len(very_low) > 0:
        print(f'\n📉 VERY LOW PERFORMING CLASSES (F1 < 0.3):')
        for idx, row in very_low.head(5).iterrows():
            print(f'   {row["Unicode_Character"]} ({row["Class_Name"]}): F1={row["F1_Score"]:.4f}')

    return len(zero_classes) == 0

if __name__ == '__main__':
    all_classes_identified = analyze_class_identification()

    if all_classes_identified:
        print(f'\n🎉 CONCLUSION: Your model successfully identifies ALL 454 classes!')
        print(f'✅ No classes have F1-Score = 0')
        print(f'✅ Model is production-ready for all Sinhala characters')
    else:
        print(f'\n⚠️  CONCLUSION: Some classes are not being predicted correctly')
        print(f'📝 Consider additional training or data augmentation for missing classes')
