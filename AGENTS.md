# dollhouse

Turns a single photo of a room into a 3D scene: it finds the objects, makes a standalone image and
then a textured mesh of each, and places the meshes where the objects stand in the photo. A Gradio
app exposes the individual steps for trying them out. Research project at the Visual Computing & AI
Lab (Prof. Nießner), TUM.

## Documentation

`documentation/` holds project documentation such as update presentations, reports and proposals.
Update presentations are in `documentation/update_presentations/`, one folder per date, with the
LaTeX template in `template/`.

`documentation/pipeline_flow.html` is an interactive flow diagram of the pipeline (stages, models,
settings and verbatim prompts, read from `src/dollhouse/`, excluding the Gradio app). Keep it up to
date: whenever a change to the pipeline adds, removes or reorders a stage, or changes a model id,
prompt or setting shown in a node panel, update the diagram in the same commit.

## Git

- One-line commit messages only — no body text.
