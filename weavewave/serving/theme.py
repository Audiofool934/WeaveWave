"""Ocean-themed Gradio UI theme for WeaveWave."""

import gradio as gr

css = """
.center-radio {
    display: flex;
    justify-content: center;
    align-items: center;
}
.compact-radio {
    width: 200px;
}
"""


def theme() -> gr.themes.Default:
    """Return an ocean-inspired Gradio theme."""
    return gr.themes.Default().set(
        # Body
        body_background_fill="linear-gradient(to bottom, #006994, #00223D)",
        body_background_fill_dark="linear-gradient(to bottom, #006994, #00223D)",
        body_text_color="#1A2430",
        body_text_color_dark="#E6F2FF",
        body_text_size="16px",
        body_text_color_subdued="#758596",
        body_text_color_subdued_dark="#A0B0C0",
        body_text_weight="400",
        embed_radius="8px",
        # Element colours
        background_fill_primary="#FFFFFF",
        background_fill_primary_dark="#283442",
        background_fill_secondary="#F2F8FF",
        background_fill_secondary_dark="#364250",
        border_color_accent="#4682B4",
        border_color_accent_dark="#6A9ACD",
        border_color_accent_subdued="#ADD8E6",
        border_color_accent_subdued_dark="#87CEFA",
        border_color_primary="#D0E0F0",
        border_color_primary_dark="#506070",
        color_accent="#29ABE2",
        color_accent_soft="#87CEEB",
        color_accent_soft_dark="#4682B4",
        # Text
        link_text_color="#0077CC",
        link_text_color_dark="#41A0FF",
        link_text_color_active="#005580",
        link_text_color_active_dark="#2980B9",
        link_text_color_hover="#00A0E9",
        link_text_color_hover_dark="#6AA2E8",
        link_text_color_visited="#551A8B",
        link_text_color_visited_dark="#8A5ACF",
        prose_text_size="16px",
        prose_text_weight="400",
        prose_header_text_weight="600",
        code_background_fill="#F0F8FF",
        code_background_fill_dark="#303A48",
        # Shadows
        shadow_drop="0 2px 4px rgba(0, 0, 0, 0.1)",
        shadow_drop_lg="0 4px 8px rgba(0, 0, 0, 0.1)",
        shadow_inset="inset 0 2px 4px rgba(0, 0, 0, 0.1)",
        shadow_spread="0 0 8px rgba(0, 0, 0, 0.1)",
        shadow_spread_dark="0 0 8px rgba(255, 255, 255, 0.05)",
        # Primary buttons
        button_primary_background_fill="#29ABE2",
        button_primary_background_fill_dark="#4682B4",
        button_primary_background_fill_hover="#1E88E5",
        button_primary_background_fill_hover_dark="#3070A0",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_primary_border_color="#29ABE2",
        button_primary_border_color_dark="#4682B4",
        button_primary_border_color_hover="#1E88E5",
        button_primary_border_color_hover_dark="#3070A0",
        button_primary_text_color_hover="#FFFFFF",
        button_primary_text_color_hover_dark="#FFFFFF",
        # Cancel buttons
        button_cancel_background_fill="#960018",
        button_cancel_background_fill_dark="#960018",
        button_cancel_background_fill_hover="#800000",
        button_cancel_background_fill_hover_dark="#800000",
        button_cancel_border_color="#960018",
        button_cancel_border_color_dark="#960018",
    )
