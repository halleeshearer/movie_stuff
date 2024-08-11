import seaborn as sns
from faicons import icon_svg
import hcp_utils as hcp
import nilearn as nl
from nilearn import plotting as plot
import shinywidgets
from shinywidgets import output_widget, render_widget

# Import data from shared.py
from shared import app_dir, df

from shiny import App, reactive, render, ui

import hcp_utils as hcp
import nilearn.plotting as plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shinywidgets import render_plotly
from icons import question_circle_fill

# import data from all_rois_mv_results
data = pd.read_csv("all_rois_mv_results.csv")

glasser = hcp.mmp.labels
# delete the first key in the dictionary
del glasser[0]
glasser_int_keys = {int(key): value for key, value in glasser.items()}
options = {"All": "All Regions"}
options.update(glasser_int_keys)

# ui.card(
#         ui.card_header(
#             ui.tooltip(
#                 ui.span("Card title ", question_circle_fill),
#                 "Additional info",
#                 placement="right",
#                 id="card_tooltip",
#             ),
#         ),
#         "Card body content...",
#     ),

app_ui = ui.page_sidebar(
    
    ui.sidebar(
        # ui.input_select(
        #     "measure",
        #     "Reliability measure",
        #     {"discr": "Discriminability", "i2c2" : "I2C2", "finger" : "Fingerprinting"},
        #     selected=["i2c2"],
        # ),
        ui.input_select(
            "plot",
            ui.tooltip(ui.span("Plot ", question_circle_fill), "Choose to plot the reliability of movie-watching, resting-state, or the difference between movie and rest."),
            {"m" : "Movie", "r" : "Rest", "diff" : "Movie-Rest Difference", "m_better" : "Significant Differences"},
            selected = ["diff"],
        ),
                
        ui.input_select(
            "roi",
            ui.tooltip(ui.span("Region of interest (Glasser parcel) ", question_circle_fill),"Show only a region of interest. Select \"All Regions\" to view all regions again."),
            # use hcp.mmp.labels as the dictionary of the regions
            options,
        ),
    ),

    ui.help_text("Click and drag brains to rotate."), 

    ui.layout_column_wrap(
        ui.card(
            ui.card_header("I2C2"),
            ui.output_ui("i2c2"),
            ui.input_slider("i2c2_max", "Range to plot", min=0, max=1, value=[0.0, 0.3], step=0.05),
            height = "600px"),
        ui.card(
            ui.card_header("Discriminability"),
            ui.output_ui("discr"),
            ui.input_slider("discr_max", "Range to plot", min=0, max=1, value=[0.0, 0.3], step=0.05),
            height = "600px"),
        ui.card(
            ui.card_header("Fingerprinting"),
            ui.output_ui("finger"),
            ui.input_slider("finger_max", "Range to plot", min=0, max=1, value=[0.0, 0.3], step=0.05),
            height = "600px"),
    ),
#h1(strong("Title"), style = "font-size:500px;")
    ui.div(ui.h3("Study Summary", style = "font-size:20px"), ui.help_text("See Shearer et al. (under revision) for more details")),

    ui.accordion(
        ui.accordion_panel("Dataset", "HCP 7T release (Van Essen et al., 2013). TR = 1000 ms. N = 109."),
        ui.accordion_panel("Scans", "Two rest scans and two movie scans from separate days. Rest1 and Movie2 on the first day, Rest4 and Movie4 on the second day. Movie runs consist of three shorter movie clips (3:48 to 4:19 minutes long) separated by 20-seconds of rest, and a 1:23 validation clip (common across movie runs) at the end. Each run was 15-16 minutes long."),
        ui.accordion_panel("Participants", "Healthy young adults between the ages of 22 and 36 years old. Includes twins, siblings, and unrelated individuals."),
        ui.accordion_panel("Processing", "HCP Minimally preprocessed data (Glasser et al., 2013). Excluded participants with mean FD greater than 0.2 mm in any run. No slice time correction was performed, spatial processing was applied, motion was corrected with FLIRT-based motion correction (no motion censoring was performed), and structured artifacts were removed using ICA + FIX. Data were represented in surface space as grayordinates (cortical surface vertices and subcortical standard-space voxels)."),
        ui.accordion_panel("Matrices", "The cortex was parcellated into 360 parcels (Glasser et al., 2016), and the 19 subcortical regions extracted with the HCP minimal preprocessing pipeline were included (Glasser et al., 2013), for a total of 379 parcels. FC matrices of parcel vertices x 379 parcels were computed using Pearson correlations between paired time courses for each parcel, subject, and run."),
        ui.accordion_panel("Reliability", ui.div(ui.div("I2C2: a non-parametric, multivariate generalization of the intraclass correlation coefficient that estimates the proportion of the total variability that arises from the subject level by using a multivariate image measurement error model (Shou et al., 2013). "), ui.div("Discriminability: an estimate of how relatively similar an individual’s repeated measurements are to each other (Bridgeford et al., 2021)."),  ui.div("Fingerprinting: the accuracy of an connectome-based identification algorithm. (Finn et al., 2015)"))),
        ui.accordion_panel("Analysis", "Each reliability estimate was computed for rest and for movie across repeat runs, producing an estimate for each parcel and condition."),
        ui.accordion_panel("Statistics", "Estimates were compared statistically between Movie and Rest for each parcel with nonparametric permutation testing (shuffling condition labels, 5000 permutations), and FDR corrected across all tests."),
        id="tab", height = "0px", open = False
    ),  



    ui.include_css(app_dir / "styles.css"),
    #title="Movie Stuff",
    title =    ui.div(
        ui.h1("Movie Stuff"),
        ui.h1("Explore test-retest reliability measures between movie-watching and resting-state functional connectivity.", style = "font-size:15px"),
        ui.h4(),
        ui.help_text("Results from Shearer et al. (under revision)", style = "font-size:12px"),
        style="text-align: left; margin-bottom: 20px;"
    ),
    fillable=True,
)


def server(input, output, session):

    @reactive.calc
    def filtered_df_i2c2():
        if input.roi() == "All":
            filt_df = data[f"i2c2_{input.plot()}"]
        # if roi is selected, then set all other values to 0
        if input.roi() != "All":
            # set all values to 0 except for the value at the index of the selected roi number
            filt_df = np.zeros(len(data))
            filt_df[int(input.roi())-1] = data[f"i2c2_{input.plot()}"][int(input.roi())-1]
        return filt_df
    
    @reactive.calc
    def filtered_df_discr():
        if input.roi() == "All":
            filt_df = data[f"discr_{input.plot()}"]
        # if roi is selected, then set all other values to 0
        if input.roi() != "All":
            # set all values to 0 except for the value at the index of the selected roi number
            filt_df = np.zeros(len(data))
            filt_df[int(input.roi())-1] = data[f"discr_{input.plot()}"][int(input.roi())-1]
        return filt_df
    
    @reactive.calc
    def filtered_df_finger():
        if input.roi() == "All":
            filt_df = data[f"finger_{input.plot()}"]
        # if roi is selected, then set all other values to 0
        if input.roi() != "All":
            # set all values to 0 except for the value at the index of the selected roi number
            filt_df = np.zeros(len(data))
            filt_df[int(input.roi())-1] = data[f"finger_{input.plot()}"][int(input.roi())-1]
        return filt_df



    # @render.text
    # def bill_length():
    #     return f"{filtered_df()['bill_length_mm'].mean():.1f} mm"

    # @render.text
    # def bill_depth():
    #     return f"{filtered_df()['bill_depth_mm'].mean():.1f} mm"


    # @render.ui
    # def mv_rel():
    #     surf_plot = plotting.view_surf(hcp.mesh.inflated, hcp.cortex_data(hcp.unparcellate(filtered_df(), hcp.mmp)), cmap="viridis", bg_map=hcp.mesh.sulc, vmax = input.plot_max(), threshold = 0.00000001)
    #     html = surf_plot.get_iframe() # get_iframe() or get_standalone()
    #     return ui.HTML(html)
    

    # create reactive strings for the plot colorbar titles depending on the selected plot
    @reactive.calc
    def colorbar_title():
        if input.plot() == "m":
            return "Movie Value"
        if input.plot() == "r":
            return "Rest Value"
        if input.plot() == "diff":
            return "Movie - Rest Value"
        if input.plot() == "m_better":
            return "Significant differences"
        
    @reactive.calc
    def color_map():
        if input.plot() == "m":
            return "OrRd"
        if input.plot() == "r":
            return "PuBu"
        if input.plot() == "diff":
            return "bwr"
        if input.plot() == "m_better":
            return "bwr"
        
    @reactive.calc
    def symmetric():
        if input.plot() == "m":
            return False
        if input.plot() == "r":
            return False
        if input.plot() == "diff":
            return True
        if input.plot() == "m_better":
            return True
        
    @reactive.calc
    def vmin():
        if input.plot() == "m":
            return 0
        if input.plot() == "r":
            return 0
        if input.plot() == "diff":
            return -0.3
        if input.plot() == "m_better":
            return -1
        
    @reactive.effect
    def update_range():
        input.plot()
        if input.plot() == "m":
            ui.update_select("i2c2_max", selected=[0.05, 1])
            ui.update_select("discr_max", selected=[0.05, 1])
            ui.update_select("finger_max", selected=[0.05, 1])

        if input.plot() == "r":
            ui.update_select("i2c2_max", selected=[0.05, 1])
            ui.update_select("discr_max", selected=[0.05, 1])
            ui.update_select("finger_max", selected=[0.05, 1])

        if input.plot() == "diff":
            ui.update_select("i2c2_max", selected=[0.05, 0.3])
            ui.update_select("discr_max", selected=[0.05, 0.3])
            ui.update_select("finger_max", selected=[0.05, 0.3])

        if input.plot() == "m_better":
            ui.update_select("i2c2_max", selected=[0.05, 1])
            ui.update_select("discr_max", selected=[0.05, 1])
            ui.update_select("finger_max", selected=[0.05, 1])


    @render.ui
    def i2c2():
        surf_plot = plotting.view_surf(hcp.mesh.inflated, hcp.cortex_data(hcp.unparcellate(filtered_df_i2c2(), hcp.mmp)), cmap=color_map(), bg_map=hcp.mesh.sulc, threshold = input.i2c2_max()[0], symmetric_cmap = symmetric(), vmin = vmin(), vmax = input.i2c2_max()[1], colorbar = True, title = colorbar_title(), title_fontsize = 15)
        surf_plot.resize(300, 300)
        html = surf_plot.get_iframe() # get_iframe() or get_standalone()
        return ui.HTML(html) 

    @render.ui
    def discr():
        surf_plot = plotting.view_surf(hcp.mesh.inflated, hcp.cortex_data(hcp.unparcellate(filtered_df_discr(), hcp.mmp)), cmap=color_map(), bg_map=hcp.mesh.sulc, threshold = input.discr_max()[0], symmetric_cmap = symmetric(), vmin = vmin(), vmax = input.discr_max()[1], colorbar = True, title = colorbar_title(), title_fontsize = 15)
        surf_plot.resize(300, 300)
        html = surf_plot.get_iframe() # get_iframe() or get_standalone()
        return ui.HTML(html) 
    
    @render.ui
    def finger():
        surf_plot = plotting.view_surf(hcp.mesh.inflated, hcp.cortex_data(hcp.unparcellate(filtered_df_finger(), hcp.mmp)), cmap=color_map(), bg_map=hcp.mesh.sulc, threshold = input.finger_max()[0], symmetric_cmap = symmetric(), vmin = vmin(), vmax = input.finger_max()[1], colorbar = True, title = colorbar_title(), title_fontsize = 15)
        surf_plot.resize(300, 300)
        html = surf_plot.get_iframe() # get_iframe() or get_standalone()
        return ui.HTML(html) 
    



    # @render.data_frame
    # def summary_statistics():
    #     cols = [
    #         "species",
    #         "island",
    #         "bill_length_mm",
    #         "bill_depth_mm",
    #         "body_mass_g",
    #     ]
    #     return render.DataGrid(filtered_df()[cols], filters=True)


app = App(app_ui, server)
