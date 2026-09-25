# PRT565 Assessment 3 – Phishing Website Detection (group package)

Two presenters · 20-slide deck (18 main + 2 backup) · ≈ 11 min 25 s total · deadline **27 September 2026, 22:00 Sydney time**.

## Folder structure

```
Final/
├── START_HERE.md                         ← this file
├── 01_Submission/                        ← the ONLY things that go in the submission ZIP
│   ├── Assessment3_Report.docx           editable report (cover page 1, video link page 2)
│   ├── Assessment3_Report.pdf            export of the DOCX (re-export after filling fields)
│   └── Assessment3_Presentation.pptx     presentation with full speaker notes per slide
├── 02_Code_and_Evidence/
│   └── phishing_project/                 runnable code, data and every result/figure
│       ├── README.md                     how to run the code
│       ├── train.py                      11 ML models + EDA, significance, importance
│       ├── deep_learning_extension.py    LSTM and RNN in PyTorch (optional)
│       ├── Phishing_Lab.ipynb            executed notebook walking through everything
│       ├── run_project.py / Start_Mac.command
│       ├── requirements.txt / requirements-deep-learning.txt
│       ├── data/                         original UCI archive, ARFF and feature docs
│       └── results/                      metrics, predictions, figures, audit, tests
└── 03_Study_Only_Do_Not_Submit/          your preparation – never submit
    ├── Presentation_Rehearsal.md         word-for-word script split Presenter A / B
    ├── QA_Preparation.md                 likely marker questions and answers
    ├── Rubric_and_Requirements_Checklist.md
    └── Code_Explanation_and_Preparation.docx/.pdf   (original code guide, 4 core models)
```

## Who presents what

| Presenter | Slides | Content | Time |
|---|---|---|---|
| **A** | 1–9 | Group intro, roadmap, motivation, dataset, EDA, data quality, preprocessing, evaluation design, models → handover | ≈ 5:40 |
| **B** | 10–18 | CV selection, test results, ROC, confusion matrix, significance, deep learning, explainability, conclusion, references | ≈ 5:45 |

Each slide shows a coloured **Presenter A / Presenter B** tag. Limits: 6 min per member, 15 min per group. Slides 19–20 are backup for questions. Full script: `03_Study_Only_Do_Not_Submit/Presentation_Rehearsal.md`.

## Finish and submit (checklist)

1. **Fill every bracketed field.** Report cover: lecturer, group number, campus, both names and student IDs. Slides 1 and 2: `[Presenter A name]`, `[Presenter B name]`, `[Student ID]`, `Group [No.]`, `[Campus]` (use PowerPoint *Replace*). Missing/incomplete cover = −15%.
2. **Record** the presentation (Zoom/Teams/PowerPoint recording) with both members speaking and cameras/voices clear. Keep it under 15 minutes.
3. **Upload** to YouTube (Unlisted), Google Drive or OneDrive, set sharing to *anyone with the link can view*, and test the link in a private/incognito window.
4. **Paste the link** on report page 2, replacing the red placeholder.
5. **Export** the DOCX to PDF (File → Save As → PDF). Check page 1 = cover, page 2 = link, title, description, motivation.
6. **ZIP** `Assessment3_Report.pdf` + `Assessment3_Presentation.pptx` (the DOCX optional). Both members submit the **same ZIP** in Learnline under your group. Include the code folder only if your lecturer asks for it. Never include folder 03.
7. Extension? Add the approval screenshot at the back of the report.

## Re-running the code

See `02_Code_and_Evidence/phishing_project/README.md`. Short version on Mac: `bash Start_Mac.command` (sets up and opens the notebook). The saved results are already complete, so you only need to re-run to demonstrate reproducibility.
