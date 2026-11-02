"""
Regression test for the Table VI bug: hardness must match the official
Spider classifier. Expected labels below are the official ones for these
Spider-dev queries (computed with taoyds/test-suite-sql-eval).

Before the fix, sql_parser returned orderBy=('asc', []) for queries without
ORDER BY, so every query got +1 on component1 and 'easy' WHERE-queries were
reported as 'medium' (dev distribution 116/358/293/267 vs 248/446/174/166).
"""
import pytest

from utils.sql_schema import Schema
from src.data.sql_parser import parse_sql
from src.evaluation.hardness import eval_hardness, get_keywords

SCHEMA = Schema({
    "singer": ["singer_id", "name", "country", "song_name", "song_release_year", "age", "is_male"],
    "stadium": ["stadium_id", "location", "name", "capacity", "highest", "lowest", "average"],
    "concert": ["concert_id", "concert_name", "theme", "stadium_id", "year"],
    "singer_in_concert": ["concert_id", "singer_id"],
})

CASES = [
    ("select count(*) from singer", "easy"),
    ("select name , country , age from singer order by age desc", "medium"),
    ("select avg(age) , min(age) , max(age) from singer where country = 'France'", "medium"),
    ("select song_name , song_release_year from singer order by age limit 1", "medium"),
    ("select distinct country from singer where age > 20", "easy"),
    ("select country , count(*) from singer group by country", "medium"),
    ("select name from stadium where capacity > 5000", "easy"),
    ("select t2.name , count(*) from concert as t1 join stadium as t2 "
     "on t1.stadium_id = t2.stadium_id group by t1.stadium_id", "medium"),
    ("select name from stadium where stadium_id not in (select stadium_id from concert)", "hard"),
    ("select t2.name , t2.location from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id "
     "where t1.year = 2014 intersect select t2.name , t2.location from concert as t1 join stadium as t2 "
     "on t1.stadium_id = t2.stadium_id where t1.year = 2015", "extra"),
]


@pytest.mark.parametrize("sql,expected", CASES)
def test_hardness_matches_official(sql, expected):
    assert eval_hardness(parse_sql(sql, SCHEMA)) == expected


def test_no_order_by_is_empty_list():
    parsed = parse_sql("select name from singer where age > 20", SCHEMA)
    assert parsed["orderBy"] == []
    kw = get_keywords(parsed)
    assert "order" not in kw and "asc" not in kw


def test_order_by_still_parsed():
    parsed = parse_sql("select name from singer order by age desc", SCHEMA)
    direction, units = parsed["orderBy"]
    assert direction == "desc" and len(units) == 1