import gradio as gr

css = """
.center-radio {
    display: flex;
    justify-content: center;
    align-items: center;
}
.compact-radio {
    width: 200px;  # 调整宽度
}
"""


def theme():
    return gr.themes.Default().set(
        # Body Attributes
        body_background_fill="linear-gradient(to bottom, #006994, #00223D)",  # Light blue, reminiscent of shallow water #E6F2FF
        body_background_fill_dark="linear-gradient(to bottom, #006994, #00223D)",  # Darker blue for dark mode #1A2430
        body_text_color="#1A2430",  # Dark blue/grey for contrast
        body_text_color_dark="#E6F2FF",  # Light blue for contrast in dark mode
        body_text_size="16px",
        body_text_color_subdued="#758596",  # Greyish blue for less important text
        body_text_color_subdued_dark="#A0B0C0",  # Lighter greyish blue in dark mode
        body_text_weight="400",
        embed_radius="8px",
        # Element Colors
        background_fill_primary="#FFFFFF",  # White background for main content areas
        background_fill_primary_dark="#283442",  # Darker background in dark mode
        background_fill_secondary="#F2F8FF",  # Slightly off-white for layered elements
        background_fill_secondary_dark="#364250",  # Darker off-white in dark mode
        border_color_accent="#4682B4",  # Steel blue for accents
        border_color_accent_dark="#6A9ACD",  # Lighter steel blue in dark mode
        border_color_accent_subdued="#ADD8E6",  # Light blue, more subdued accent
        border_color_accent_subdued_dark="#87CEFA",  # Lighter blue, more subdued accent in dark mode
        border_color_primary="#D0E0F0",  # Light greyish blue for borders
        border_color_primary_dark="#506070",  # Darker greyish blue for dark mode
        color_accent="#29ABE2",  # Bright blue for highlights
        color_accent_soft="#87CEEB",  # Sky blue, softer accent
        color_accent_soft_dark="#4682B4",  # Steel blue, softer accent in dark mode
        # Text
        link_text_color="#0077CC",  # Standard blue link color
        link_text_color_dark="#41A0FF",  # Lighter blue link in dark mode
        link_text_color_active="#005580",  # Darker blue when link is active
        link_text_color_active_dark="#2980B9",  # Slightly darker blue when active in dark mode
        link_text_color_hover="#00A0E9",  # Brighter blue on hover
        link_text_color_hover_dark="#6AA2E8",  # Lighter brighter blue on hover in dark mode
        link_text_color_visited="#551A8B",  # Purple for visited links (adjust as desired)
        link_text_color_visited_dark="#8A5ACF",  # Lighter purple for visited links in dark mode
        prose_text_size="16px",
        prose_text_weight="400",
        prose_header_text_weight="600",
        code_background_fill="#F0F8FF",  # Very light blue for code blocks
        code_background_fill_dark="#303A48",  # Darker blue for code blocks in dark mode
        # Shadows
        shadow_drop="0 2px 4px rgba(0, 0, 0, 0.1)",
        shadow_drop_lg="0 4px 8px rgba(0, 0, 0, 0.1)",
        shadow_inset="inset 0 2px 4px rgba(0, 0, 0, 0.1)",
        shadow_spread="0 0 8px rgba(0, 0, 0, 0.1)",
        shadow_spread_dark="0 0 8px rgba(255, 255, 255, 0.05)",
        # ... (Rest of the parameters -  apply similar ocean-themed color choices)
        # Example for buttons:
        button_primary_background_fill="#29ABE2",  # Bright blue for primary buttons
        button_primary_background_fill_dark="#4682B4",  # Steel blue in dark mode
        button_primary_background_fill_hover="#1E88E5",  # Slightly darker blue on hover
        button_primary_background_fill_hover_dark="#3070A0",  # Slightly darker in dark mode
        button_primary_text_color="#FFFFFF",  # White text on blue buttons
        button_primary_text_color_dark="#FFFFFF",  # White text in dark mode
        button_primary_border_color="#29ABE2",
        button_primary_border_color_dark="#4682B4",
        button_primary_border_color_hover="#1E88E5",
        button_primary_border_color_hover_dark="#3070A0",
        button_primary_text_color_hover="#FFFFFF",
        button_primary_text_color_hover_dark="#FFFFFF",
        # ... (Continue for other components)
        button_cancel_background_fill="#960018",
        button_cancel_background_fill_dark="#960018",
        button_cancel_background_fill_hover="#800000",
        button_cancel_background_fill_hover_dark="#800000",
        button_cancel_border_color="#960018",
        button_cancel_border_color_dark="#960018",
    )
