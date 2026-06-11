"""Unit tests for system/user prompt splitting.

Covers parse_system_user, parse_template_variables (XML-aware),
render_template, and _normalize_escapes — all part of the
feature/system-user-prompt-splitting convention.
"""

import pytest
from server.templates import (
    parse_system_user,
    parse_template_variables,
    render_template,
    _normalize_escapes,
)


# ---------------------------------------------------------------------------
# parse_system_user — plain strings
# ---------------------------------------------------------------------------

class TestParseSystemUserPlain:
    def test_plain_string_returns_none_system(self):
        system, user = parse_system_user("Hello, {{name}}!")
        assert system is None
        assert user == "Hello, {{name}}!"

    def test_empty_string_returns_none_system(self):
        system, user = parse_system_user("")
        assert system is None
        assert user == ""

    def test_plain_multiline_string_unchanged(self):
        template = "Line 1\nLine 2\nLine 3"
        system, user = parse_system_user(template)
        assert system is None
        assert user == template

    def test_unknown_xml_tags_treated_as_plain(self):
        template = "<prompt>You are a helpful assistant.</prompt>"
        system, user = parse_system_user(template)
        assert system is None
        assert user == template

    def test_content_with_angle_brackets_not_our_tags(self):
        """Angle brackets used as comparison operators are not confused with XML tags."""
        template = "Filter items where x > 0 and y < 10."
        system, user = parse_system_user(template)
        assert system is None
        assert user == template


# ---------------------------------------------------------------------------
# parse_system_user — XML format with <system> and/or <user> tags
# ---------------------------------------------------------------------------

class TestParseSystemUserXml:
    def test_xml_system_and_user_tags(self):
        template = (
            "<system>\nYou are a helpful assistant.\n</system>\n\n"
            "<user>\nHelp me with {{topic}}.\n</user>"
        )
        system, user = parse_system_user(template)
        assert system == "You are a helpful assistant."
        assert user == "Help me with {{topic}}."

    def test_xml_strips_leading_trailing_whitespace(self):
        template = (
            "<system>\n\n  System text.  \n\n</system>\n\n"
            "<user>\n\n  User text.  \n\n</user>"
        )
        system, user = parse_system_user(template)
        assert system == "System text."
        assert user == "User text."

    def test_xml_system_only_no_user_tag(self):
        """Content outside <system> becomes the user portion."""
        template = "<system>\nSystem content.\n</system>\n\nUser content here."
        system, user = parse_system_user(template)
        assert system == "System content."
        assert user == "User content here."

    def test_xml_user_only_no_system_tag(self):
        """Standalone <user> tag with no <system> returns (None, user_content)."""
        template = "<user>\nJust user content.\n</user>"
        system, user = parse_system_user(template)
        assert system is None
        assert user == "Just user content."

    def test_xml_multiline_content_in_both_sections(self):
        template = (
            "<system>\n"
            "You are a helpful assistant.\n"
            "Always be polite.\n"
            "Never make things up.\n"
            "</system>\n\n"
            "<user>\n"
            "Line A\n"
            "Line B\n"
            "</user>"
        )
        system, user = parse_system_user(template)
        assert "You are a helpful assistant." in system
        assert "Always be polite." in system
        assert "Line A" in user
        assert "Line B" in user

    def test_xml_variables_in_both_sections(self):
        template = (
            "<system>\nYou assist {{company}}.\n</system>\n\n"
            "<user>\nHelp with {{topic}}.\n</user>"
        )
        system, user = parse_system_user(template)
        assert "{{company}}" in system
        assert "{{topic}}" in user

    def test_xml_variable_in_system_not_leaked_to_user(self):
        template = (
            "<system>\nAssistant for {{company}}.\n</system>\n\n"
            "<user>\nWhat can you help with?\n</user>"
        )
        system, user = parse_system_user(template)
        assert "{{company}}" in system
        assert "{{company}}" not in user

    def test_xml_content_with_angle_brackets_in_user(self):
        """Angle brackets inside the user section (code snippets, comparisons) are preserved."""
        template = (
            "<system>\nYou are a code reviewer.\n</system>\n\n"
            "<user>\nReview this: if x > 0 and y < 10.\n</user>"
        )
        system, user = parse_system_user(template)
        assert system == "You are a code reviewer."
        assert "x > 0" in user
        assert "y < 10" in user

    def test_xml_special_regex_chars_in_content(self):
        """Regex special characters in content don't break parsing."""
        template = (
            "<system>\nMatch *.txt files.\n</system>\n\n"
            "<user>\nPattern: ^[a-z]+$\n</user>"
        )
        system, user = parse_system_user(template)
        assert "*.txt" in system
        assert "^[a-z]+$" in user

    def test_xml_empty_system_tag_returns_empty_string(self):
        """An empty <system></system> tag yields system='', not None.

        In the run route, empty strings are falsy and treated as no system prompt,
        so this edge case is handled consistently downstream.
        """
        template = "<system></system>\n\n<user>User content.</user>"
        system, user = parse_system_user(template)
        assert system == ""
        assert user == "User content."

    def test_xml_unclosed_system_tag_falls_back_to_plain(self):
        """An unclosed <system> tag has no regex match — treated as plain string."""
        template = "<system>This is never closed. {{var}}"
        system, user = parse_system_user(template)
        assert system is None
        assert user == template

    def test_xml_real_world_multi_variable_prompt(self):
        """System/user XML with several template variables (typical registry prompt)."""
        template = (
            "<system>\n"
            "You are a helpful assistant for {{company}}.\n"
            "Your tone is warm but direct.\n"
            "</system>\n\n"
            "<user>\n"
            "Question: {{prompt_question}}\n"
            "Current answer: {{current_answer}}\n"
            "Tone: {{tone}}\n"
            "</user>"
        )
        system, user = parse_system_user(template)
        assert "{{company}}" in system
        assert "helpful assistant" in system
        assert "{{prompt_question}}" in user
        assert "{{current_answer}}" in user
        assert "{{tone}}" in user
        assert "<system>" not in system
        assert "<user>" not in user


# ---------------------------------------------------------------------------
# parse_system_user — list[dict] native MLflow chat format
# ---------------------------------------------------------------------------

class TestParseSystemUserListFormat:
    def test_list_with_system_and_user(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        system, user = parse_system_user(msgs)
        assert system == "You are helpful."
        assert user == "Hello!"

    def test_list_user_only(self):
        msgs = [{"role": "user", "content": "Just the user."}]
        system, user = parse_system_user(msgs)
        assert system is None
        assert user == "Just the user."

    def test_list_multiple_user_messages_joined(self):
        msgs = [
            {"role": "system", "content": "System here."},
            {"role": "user", "content": "First user turn."},
            {"role": "user", "content": "Second user turn."},
        ]
        system, user = parse_system_user(msgs)
        assert system == "System here."
        assert user == "First user turn.\n\nSecond user turn."

    def test_list_empty_returns_none_and_empty_string(self):
        system, user = parse_system_user([])
        assert system is None
        assert user == ""

    def test_list_missing_content_key_handled(self):
        msgs = [{"role": "user"}]
        system, user = parse_system_user(msgs)
        assert system is None
        assert user == ""

    def test_list_unknown_role_ignored(self):
        msgs = [
            {"role": "system", "content": "System."},
            {"role": "assistant", "content": "This is ignored."},
            {"role": "user", "content": "User."},
        ]
        system, user = parse_system_user(msgs)
        assert system == "System."
        assert user == "User."

    def test_list_system_only_no_user(self):
        msgs = [{"role": "system", "content": "System only."}]
        system, user = parse_system_user(msgs)
        assert system == "System only."
        assert user == ""


# ---------------------------------------------------------------------------
# parse_template_variables — XML-aware extraction
# ---------------------------------------------------------------------------

class TestParseTemplateVariablesXml:
    def test_variables_from_both_sections(self):
        template = (
            "<system>\nYou assist {{company}}.\n</system>\n\n"
            "<user>\nHelp with {{topic}}.\n</user>"
        )
        vars_ = parse_template_variables(template)
        assert "company" in vars_
        assert "topic" in vars_

    def test_variables_only_in_system_section(self):
        template = (
            "<system>\nAssistant for {{company}}.\n</system>\n\n"
            "<user>\nWhat can you help with?\n</user>"
        )
        vars_ = parse_template_variables(template)
        assert "company" in vars_
        assert len(vars_) == 1

    def test_variables_only_in_user_section(self):
        template = (
            "<system>\nYou are a helpful assistant.\n</system>\n\n"
            "<user>\nHelp me with {{topic}} and {{subtopic}}.\n</user>"
        )
        vars_ = parse_template_variables(template)
        assert vars_ == ["topic", "subtopic"]

    def test_variables_deduplicated_across_sections(self):
        """Same variable in both system and user appears only once."""
        template = (
            "<system>\nYou work for {{company}}.\n</system>\n\n"
            "<user>\nTell me about {{company}}.\n</user>"
        )
        vars_ = parse_template_variables(template)
        assert vars_.count("company") == 1

    def test_variables_order_preserved_system_first(self):
        """Variables from system appear before variables from user in the result."""
        template = (
            "<system>\nYou are {{role}}.\n</system>\n\n"
            "<user>\nMy name is {{name}}. I need {{service}}.\n</user>"
        )
        vars_ = parse_template_variables(template)
        assert vars_ == ["role", "name", "service"]

    def test_variables_from_list_format(self):
        msgs = [
            {"role": "system", "content": "You are {{role}}."},
            {"role": "user", "content": "Help with {{topic}}."},
        ]
        vars_ = parse_template_variables(msgs)
        assert "role" in vars_
        assert "topic" in vars_


# ---------------------------------------------------------------------------
# render_template — substitution safety
# ---------------------------------------------------------------------------

class TestRenderTemplate:
    def test_renders_user_template(self):
        assert render_template("Help me with {{topic}}.", {"topic": "Python"}) == "Help me with Python."

    def test_renders_system_template(self):
        assert render_template(
            "You are an assistant for {{company}}.", {"company": "Acme"}
        ) == "You are an assistant for Acme."

    def test_missing_variable_leaves_placeholder(self):
        result = render_template("Help me with {{topic}}.", {})
        assert result == "Help me with {{topic}}."

    def test_extra_variables_are_ignored(self):
        result = render_template("Hello {{name}}.", {"name": "Alice", "unused": "x"})
        assert result == "Hello Alice."

    def test_xml_template_preserves_xml_structure(self):
        """Rendering substitutes variables but keeps XML tags intact."""
        template = (
            "<system>\nYou assist {{company}}.\n</system>\n\n"
            "<user>\nHelp with {{topic}}.\n</user>"
        )
        rendered = render_template(template, {"company": "Acme", "topic": "billing"})
        assert "You assist Acme." in rendered
        assert "Help with billing." in rendered
        assert "<system>" in rendered
        assert "<user>" in rendered

    def test_variable_value_with_backslash_does_not_crash(self):
        """Variable values containing backslashes (e.g. Windows paths) are rendered literally."""
        result = render_template("Path: {{path}}", {"path": r"C:\Users\john\documents"})
        assert r"C:\Users\john\documents" in result

    def test_variable_value_with_backslash_digit_does_not_crash(self):
        """Variable values containing \\1, \\2 etc. are rendered literally, not as regex backrefs."""
        result = render_template("Value: {{val}}", {"val": r"\1 hello"})
        assert r"\1 hello" in result

    def test_variable_value_with_special_regex_chars(self):
        """Dollar signs, parens, and other regex chars in values are treated as literals."""
        result = render_template("Price: {{price}}", {"price": "$100.00 (USD)"})
        assert result == "Price: $100.00 (USD)"

    def test_variable_value_with_xml_like_content(self):
        """XML-like content in a variable value is rendered as-is (not re-parsed)."""
        result = render_template("Content: {{data}}", {"data": "<b>bold</b>"})
        assert result == "Content: <b>bold</b>"

    def test_sql_injection_like_value_is_rendered_safely(self):
        """SQL injection patterns in variable values are rendered as literal text."""
        result = render_template("Name: {{name}}", {"name": "'; DROP TABLE prompts; --"})
        assert "DROP TABLE" in result
        # Just a string — it never reaches a database

    def test_render_multiple_occurrences(self):
        result = render_template("{{x}} and {{x}} again.", {"x": "hello"})
        assert result == "hello and hello again."


# ---------------------------------------------------------------------------
# _normalize_escapes
# ---------------------------------------------------------------------------

class TestNormalizeEscapes:
    def test_literal_backslash_n_becomes_newline(self):
        assert _normalize_escapes("line1\\nline2") == "line1\nline2"

    def test_literal_backslash_t_becomes_tab(self):
        assert _normalize_escapes("col1\\tcol2") == "col1\tcol2"

    def test_real_newline_is_unchanged(self):
        text = "line1\nline2"
        assert _normalize_escapes(text) == text

    def test_real_tab_is_unchanged(self):
        text = "col1\tcol2"
        assert _normalize_escapes(text) == text

    def test_mixed_literal_and_real_escapes(self):
        result = _normalize_escapes("a\\nb\\tc")
        assert result == "a\nb\tc"

    def test_empty_string(self):
        assert _normalize_escapes("") == ""

    def test_no_escapes(self):
        text = "plain text with no escapes"
        assert _normalize_escapes(text) == text

    def test_multiple_literal_newlines(self):
        result = _normalize_escapes("a\\nb\\nc")
        assert result == "a\nb\nc"
