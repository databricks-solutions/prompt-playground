"""Tests for prompt template variable rendering."""

from server.templates import render_template


def test_basic_substitution():
    template = "Hello, {{name}}!"
    result = render_template(template, {"name": "Alice"})
    assert result == "Hello, Alice!"


def test_multiple_variables():
    template = "{{greeting}}, {{name}}. Your score is {{score}}."
    result = render_template(template, {"greeting": "Hi", "name": "Bob", "score": "95"})
    assert result == "Hi, Bob. Your score is 95."


def test_repeated_variable_replaced_everywhere():
    template = "{{name}} said hello. {{name}} said goodbye."
    result = render_template(template, {"name": "Carol"})
    assert result == "Carol said hello. Carol said goodbye."


def test_extra_variables_ignored():
    template = "Hello, {{name}}!"
    result = render_template(template, {"name": "Dave", "unused": "value"})
    assert result == "Hello, Dave!"


def test_missing_variable_left_as_placeholder():
    template = "Hello, {{name}}! Your role is {{role}}."
    result = render_template(template, {"name": "Eve"})
    assert result == "Hello, Eve! Your role is {{role}}."


def test_empty_variables_dict():
    template = "Hello, {{name}}!"
    result = render_template(template, {})
    assert result == "Hello, {{name}}!"


def test_variables_with_spaces_in_template():
    template = "Hello, {{ name }}!"
    result = render_template(template, {"name": "Frank"})
    assert result == "Hello, Frank!"


def test_multiline_template():
    template = """You are a {{role}} assistant.

The user asked: {{question}}

Please respond helpfully."""
    result = render_template(template, {"role": "helpful", "question": "What is 2+2?"})
    assert "helpful" in result
    assert "What is 2+2?" in result
    assert "{{role}}" not in result
    assert "{{question}}" not in result


def test_special_regex_characters_in_value():
    # Values containing regex special chars should be treated literally
    template = "Pattern: {{pattern}}"
    result = render_template(template, {"pattern": "^[a-z]+$"})
    assert result == "Pattern: ^[a-z]+$"
