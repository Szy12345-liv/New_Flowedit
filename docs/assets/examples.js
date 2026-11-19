// Generated from images/examples.json by tools/sync_site_data.py.
window.DECFLOWEDIT_EXAMPLES = {
  "dataset": "PIE-Bench",
  "paper_figure": "Figure 1",
  "backbones": [
    {
      "id": "flux",
      "label": "FLUX.1-dev",
      "methods": [
        {
          "id": "ddib",
          "label": "DDIB"
        },
        {
          "id": "flowedit",
          "label": "FlowEdit"
        },
        {
          "id": "decflowedit",
          "label": "DecFlowEdit"
        }
      ]
    },
    {
      "id": "sd3",
      "label": "Stable Diffusion 3",
      "methods": [
        {
          "id": "ddib",
          "label": "DDIB"
        },
        {
          "id": "flowedit",
          "label": "FlowEdit"
        },
        {
          "id": "flowalign",
          "label": "FlowAlign"
        },
        {
          "id": "splitflow",
          "label": "SplitFlow"
        },
        {
          "id": "decflowedit",
          "label": "DecFlowEdit"
        }
      ]
    }
  ],
  "examples": [
    {
      "id": "cake",
      "sample_id": "000000000001",
      "title": "Shape change",
      "source_prompt": "A round cake on a wooden plate",
      "target_prompt": "A square cake on a wooden plate",
      "source": "assets/images/examples/cake/source.png",
      "images": {
        "flux": {
          "ddib": "assets/images/examples/cake/ddib-flux.png",
          "flowedit": "assets/images/examples/cake/flowedit-flux.png",
          "decflowedit": "assets/images/examples/cake/decflowedit-flux.png"
        },
        "sd3": {
          "ddib": "assets/images/examples/cake/ddib-sd3.png",
          "flowedit": "assets/images/examples/cake/flowedit-sd3.png",
          "flowalign": "assets/images/examples/cake/flowalign-sd3.png",
          "splitflow": "assets/images/examples/cake/splitflow-sd3.png",
          "decflowedit": "assets/images/examples/cake/decflowedit-sd3.png"
        }
      }
    },
    {
      "id": "squirrel",
      "sample_id": "221000000008",
      "title": "Object addition",
      "source_prompt": "A squirrel sitting on a wooden platform",
      "target_prompt": "A squirrel sitting on a wooden platform reading a book",
      "source": "assets/images/examples/squirrel/source.png",
      "images": {
        "flux": {
          "ddib": "assets/images/examples/squirrel/ddib-flux.png",
          "flowedit": "assets/images/examples/squirrel/flowedit-flux.png",
          "decflowedit": "assets/images/examples/squirrel/decflowedit-flux.png"
        },
        "sd3": {
          "ddib": "assets/images/examples/squirrel/ddib-sd3.png",
          "flowedit": "assets/images/examples/squirrel/flowedit-sd3.png",
          "flowalign": "assets/images/examples/squirrel/flowalign-sd3.png",
          "splitflow": "assets/images/examples/squirrel/splitflow-sd3.png",
          "decflowedit": "assets/images/examples/squirrel/decflowedit-sd3.png"
        }
      }
    },
    {
      "id": "kitten",
      "sample_id": "321000000003",
      "title": "Object removal",
      "source_prompt": "A kitten playing with balls",
      "target_prompt": "A kitten",
      "source": "assets/images/examples/kitten/source.png",
      "images": {
        "flux": {
          "ddib": "assets/images/examples/kitten/ddib-flux.png",
          "flowedit": "assets/images/examples/kitten/flowedit-flux.png",
          "decflowedit": "assets/images/examples/kitten/decflowedit-flux.png"
        },
        "sd3": {
          "ddib": "assets/images/examples/kitten/ddib-sd3.png",
          "flowedit": "assets/images/examples/kitten/flowedit-sd3.png",
          "flowalign": "assets/images/examples/kitten/flowalign-sd3.png",
          "splitflow": "assets/images/examples/kitten/splitflow-sd3.png",
          "decflowedit": "assets/images/examples/kitten/decflowedit-sd3.png"
        }
      }
    },
    {
      "id": "new-york",
      "sample_id": "822000000002",
      "title": "Background change",
      "source_prompt": "A woman on top of a mountain",
      "target_prompt": "A woman in New York",
      "source": "assets/images/examples/new-york/source.png",
      "images": {
        "flux": {
          "ddib": "assets/images/examples/new-york/ddib-flux.png",
          "flowedit": "assets/images/examples/new-york/flowedit-flux.png",
          "decflowedit": "assets/images/examples/new-york/decflowedit-flux.png"
        },
        "sd3": {
          "ddib": "assets/images/examples/new-york/ddib-sd3.png",
          "flowedit": "assets/images/examples/new-york/flowedit-sd3.png",
          "flowalign": "assets/images/examples/new-york/flowalign-sd3.png",
          "splitflow": "assets/images/examples/new-york/splitflow-sd3.png",
          "decflowedit": "assets/images/examples/new-york/decflowedit-sd3.png"
        }
      }
    }
  ]
};
