# -*- coding: utf-8 -*-
"""HTML 行业垂直新闻源解析器

为 IntelligenceService 的 source_type=html 资讯源提供解析能力：
抓取到的列表页 HTML 按站点配置（html_news_sources.yaml）解析出
标题 / 链接 / 发布时间，落库 intelligence_items 后随 news_context
进入个股分析 prompt。

设计：
- 新增站点只需在 html_news_sources.yaml 加一段配置，无需改代码
- 解析器三类：rss（正则轻实现）/ jsonld（页面内嵌 JSON-LD）/ regex（YAML 正则）
- 日期归一化支持「11月5日」、RFC2822、HH:MM、N小时前、YYYY-MM 等站点格式
- 发布时间缺失时回退为抓取时间（列表页条目均为近期内容），
  避免 NULL published_at 被 list_items 的时间窗过滤掉
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "html_news_sources.yaml"

_MONTH_DAY_RE = re.compile(r"^(\d{1,2})月(\d{1,2})日$")
_RFC822_MONTH = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_config_cache: Optional[List[Dict[str, Any]]] = None
_config_lock = threading.Lock()


def _load_config() -> List[Dict[str, Any]]:
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    with _config_lock:
        if _config_cache is None:
            try:
                import yaml
                data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
                _config_cache = [s for s in data.get("sources", []) if isinstance(s, dict) and s.get("url")]
            except Exception as exc:  # 配置缺失/损坏时降级为无 html 规则
                logger.warning("html_news_sources.yaml 加载失败: %s", exc)
                _config_cache = []
    return _config_cache


def reset_config_cache() -> None:
    """配置文件变更后重载（主要供测试使用）"""
    global _config_cache
    with _config_lock:
        _config_cache = None


def get_source_config(url: str) -> Optional[Dict[str, Any]]:
    """按 URL 精确匹配或同域匹配查找站点解析配置"""
    url = (url or "").strip()
    if not url:
        return None
    sources = _load_config()
    for src in sources:
        if str(src.get("url", "")).strip() == url:
            return src
    try:
        from urllib.parse import urlparse

        host = urlparse(url).hostname or ""
    except ValueError:
        return None
    if not host:
        return None
    for src in sources:
        try:
            if (urlparse(str(src.get("url", ""))).hostname or "") == host:
                return src
        except ValueError:
            continue
    return None


def _decode_html(content: bytes, encoding: str) -> str:
    if encoding:
        try:
            return content.decode(encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            pass
    for enc in ("utf-8", "gb18030"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _norm_date(raw: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """归一化站点日期文本；无法解析时返回 None（由调用方回退为抓取时间）"""
    now = now or datetime.now()
    raw = (raw or "").strip()

    # RFC2822（RSS pubDate）: Fri, 06 Nov 2026 13:50:00 GMT
    m = re.match(r"^\w{3},?\s*(\d{1,2})\s+([A-Za-z]{3})\w*\s+(\d{4})", raw)
    if m:
        mon = _RFC822_MONTH.get(m.group(2)[:3].lower())
        if mon:
            try:
                return datetime(int(m.group(3)), mon, int(m.group(1)))
            except ValueError:
                return None
    m = _MONTH_DAY_RE.match(raw)
    if m:
        mon, day = int(m.group(1)), int(m.group(2))
        year = now.year
        # 简单跨年处理：当前月份很小而新闻月份很大 → 去年
        if now.month <= 2 and mon >= 11:
            year -= 1
        try:
            return datetime(year, mon, day)
        except ValueError:
            return None
    m = re.match(r"^(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", raw)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.match(r"^(\d{1,2})-(\d{1,2})\s+(\d{1,2}:\d{2})", raw)
    if m:
        try:
            parsed = datetime.strptime(f"{now.year}-{int(m.group(1)):02d}-{int(m.group(2)):02d} {m.group(3)}", "%Y-%m-%d %H:%M")
        except ValueError:
            return None
        if parsed > now:  # 未来时间 → 实为去年
            try:
                parsed = parsed.replace(year=now.year - 1)
            except ValueError:
                pass
        return parsed
    # 仅时分（如证券时报「20:34」）→ 今天
    m = re.match(r"^(\d{1,2}):(\d{2})$", raw)
    if m:
        try:
            return datetime(now.year, now.month, now.day, int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None
    # 相对时间（C114「1小时前」等）→ 按实际回退
    m = re.match(r"^(\d+)\s*小时前$", raw)
    if m:
        from datetime import timedelta

        return now - timedelta(hours=int(m.group(1)))
    m = re.match(r"^(\d+)\s*分钟前$", raw)
    if m:
        from datetime import timedelta

        return now - timedelta(minutes=int(m.group(1)))
    if raw in ("刚刚", "今天"):
        return now
    # 仅年月（如 OFweek URL 里的 2026-11）→ 当月 1 日
    m = re.match(r"^(20\d{2})-(\d{1,2})$", raw)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            return None
    return None


def _ld_nodes(data: Any) -> Iterator[dict]:
    """深度遍历 JSON-LD 节点：@graph / itemListElement / item / mainEntity 嵌套"""
    if isinstance(data, list):
        for x in data:
            yield from _ld_nodes(x)
    elif isinstance(data, dict):
        yield data
        for key in ("@graph", "itemListElement", "item", "mainEntity"):
            if key in data:
                yield from _ld_nodes(data[key])


def _parse_jsonld(src: dict, text: str) -> List[Dict[str, Any]]:
    items, seen = [], set()
    for m in re.finditer(r'<script type="application/ld\+json">(.*?)</script>', text, re.S):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        for d in _ld_nodes(data):
            if d.get("@type") not in ("NewsArticle", "Article", "NewsItem") or not d.get("headline"):
                continue
            url = str(d.get("url") or d.get("@id") or "").split("#")[0]
            key = (str(d["headline"])[:60], url)
            if key in seen:
                continue
            seen.add(key)
            description = str(d.get("description") or "")
            items.append({
                "title": str(d["headline"])[:200],
                "summary": description[:500],
                "url": url,
                "published": _norm_date(str(d.get("datePublished") or "")),
            })
    return items


def _parse_regex(src: dict, text: str) -> List[Dict[str, Any]]:
    fields = src.get("fields") or {}
    title_re = fields.get("title", "")
    date_re = fields.get("date", "")
    if not title_re:
        return []
    items, seen = [], set()
    base_m = re.match(r"(https?)://([^/]+)", str(src.get("url", "")))
    for m in re.finditer(title_re, text, re.S):
        title = m.group("title").strip()
        url = m.group("url").strip()
        # 相对链接 → 绝对 URL（如 stcn/wabei/yaozh）
        if base_m and not url.lower().startswith("http"):
            if url.startswith("//"):
                url = f"{base_m.group(1)}:{url}"
            elif url.startswith("/"):
                url = f"{base_m.group(1)}://{base_m.group(2)}{url}"
        if title in seen or len(title) < 6:
            continue
        seen.add(title)
        d = ""
        if date_re:
            dm = re.search(date_re, text[max(0, m.start() - 200): m.end() + 200])
            if dm:
                d = dm.group("date")
        items.append({
            "title": title[:200],
            "summary": "",
            "url": url,
            "published": _norm_date(d),
        })
    return items


def _parse_rss_text(src: dict, text: str) -> List[Dict[str, Any]]:
    items = []
    for m in re.finditer(
            r"<item>.*?<title>(?:<!\[CDATA\[)?(?P<t>.*?)(?:\]\]>)?</title>.*?"
            r"<link>(?:<!\[CDATA\[)?(?P<u>.*?)(?:\]\]>)?</link>.*?"
            r"<pubDate>(?:<!\[CDATA\[)?(?P<d>.*?)(?:\]\]>)?</pubDate>",
            text, re.S):
        items.append({
            "title": m.group("t").strip()[:200],
            "summary": "",
            "url": m.group("u").strip(),
            "published": _norm_date(m.group("d")),
        })
    if not items:  # atom
        for m in re.finditer(
                r"<entry>.*?<title>(?:<!\[CDATA\[)?(?P<t>.*?)(?:\]\]>)?</title>.*?"
                r'href="(?P<u>https?://[^"]+)".*?<updated>(?P<d>[^<]+)</updated>',
                text, re.S):
            items.append({
                "title": m.group("t").strip()[:200],
                "summary": "",
                "url": m.group("u").strip(),
                "published": _norm_date(m.group("d")),
            })
    return items


def parse_html_news(
    content: bytes,
    *,
    url: str,
    source_name: str,
    limit: int,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    解析 html 类型资讯源的列表页。

    Returns:
        最多 limit 条 {"title", "summary", "url", "published": datetime}；
        published 缺失时回退为抓取时间。站点未配置解析规则时回退
        尝试通用 JSON-LD 解析，仍无结果则返回空列表。
    """
    now = now or datetime.now()
    src = get_source_config(url)
    stype = (src or {}).get("type", "")
    text = _decode_html(content, str((src or {}).get("encoding") or ""))

    if stype == "jsonld":
        parsed = _parse_jsonld(src, text)
    elif stype == "regex":
        parsed = _parse_regex(src, text)
    elif stype == "rss":
        parsed = _parse_rss_text(src, text)
    else:
        parsed = _parse_jsonld({"name": source_name}, text) if text.lstrip().startswith("<") else []

    entries: List[Dict[str, Any]] = []
    for item in parsed:
        published = item.get("published") or now
        entries.append({
            "title": item["title"],
            "summary": item.get("summary") or "",
            "url": item["url"],
            "published": published,
        })
        if len(entries) >= max(1, int(limit)):
            break
    return entries
