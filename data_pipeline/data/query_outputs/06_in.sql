SELECT title, category_id FROM books WHERE category_id IN (SELECT category_id FROM categories WHERE category_name IN ('Travel','Mystery')) ORDER BY title;
