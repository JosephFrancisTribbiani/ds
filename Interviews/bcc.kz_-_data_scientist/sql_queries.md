Для решения задач использовался синтаксис **PostgreSQL**

### Задача 1

Напишите запрос, который выведет в разрезе стран информацию о кол-ве сотрудников, которые работают на данный момент в компании, и минимальная зарплата должности которых превышает $2000

```SQL
SELECT subquery_3.country_name, count(subquery_3.country_name) AS qty FROM (
	SELECT * FROM (
		SELECT departments.department_id, countries.country_name FROM countries
		JOIN locations ON countries.country_id = locations.country_id
		JOIN departments ON locations.location_id = departments.location_id
		) AS subquery_1
	JOIN (
		SELECT DISTINCT employees.employee_id, employees.department_id FROM employees
		JOIN jobs
		ON employees.job_id = jobs.job_id
		WHERE employees.employee_id NOT IN (SELECT job_history.employee_id FROM job_history)
		AND jobs.min_salary > 2000
		) AS subquery_2
	ON subquery_1.department_id = subquery_2.department_id
	) AS subquery_3
GROUP BY subquery_3.country_name
ORDER BY qty DESC
```


### Задача 2

 Напишите запрос, который выведет информацию о топ 10 заказчиках (Имя, Фамилия, номер телефона), которые сделали совершили больше всего заказов в текущем месяце

```SQL
SELECT customers.cust_first_name, customers.cust_last_name, customers.phone_numbers FROM (
	SELECT customer_id, count(DISTINCT order_id) as norders FROM orders
	WHERE date_trunc('month', order_date) = date_trunc('month', now())
	GROUP BY customer_id
	ORDER BY norders DESC
	) AS subquery
LEFT JOIN customers
ON subquery.customer_id = customers.customer_id
LIMIT 10
```


### Задача 3

Напишите запрос, который выведет информацию в разрезе заказчиков: Имя, Фамилия, Имя и Фамилия менеджера, Должность, Департамент, Регион, Страна, Имя и Фамилия менеджера сотрудника, кол-во совершенных заказов за последние 30 дней, кол-во совершенных заказов за последний 3 месяца, месяц с максимальным кол-вом заказов за последний год, самый популярный заказываемый продукт за последний месяц, кол-во заказов, где стоимость заказа превышает среднюю стоимость всех его заказов за последний год.

```SQL
-- создадим views для большей читаемости текста запроса
-- выведение информации по менеджерам компании
CREATE OR REPLACE VIEW employees_info_view AS
	SELECT employees.employee_id, employees.first_name, employees.last_name, jobs.job_title, 
		departments.department_name, countries.country_name, regions.region_name,
		managers.first_name AS manager_first_name, managers.last_name AS manager_last_name FROM employees
	LEFT JOIN jobs ON employees.job_id = jobs.job_id
	LEFT JOIN departments ON employees.department_id = departments.department_id
	LEFT JOIN locations ON departments.location_id = locations.location_id
	LEFT JOIN countries ON locations.country_id = countries.country_id
	LEFT JOIN regions ON countries.region_id = regions.region_id
	LEFT JOIN employees AS managers ON employees.manager_id = managers.employee_id;

-- количество заказов за последние 30 дней
CREATE OR REPLACE VIEW last_30_days_view AS
	SELECT customer_id, count(*) as last_30_days FROM orders
	WHERE order_date >= date_trunc('day', now()) - interval '29 days'
	GROUP BY customer_id;

-- количество заказов за последние 3 месяца
CREATE OR REPLACE VIEW last_3_month_view  AS
	SELECT customer_id, count(*) as last_3_month FROM orders
	WHERE order_date >= date_trunc('month', now()) - interval '2 month'
	GROUP BY customer_id;
	
--месяц с максимальным количеством заказов за последний год
CREATE OR REPLACE VIEW best_month_view AS
	SELECT DISTINCT ON (subquery.customer_id) subquery.customer_id, subquery.best_month FROM (
		SELECT customer_id, EXTRACT('month' FROM order_date) AS best_month, count(*) as qty FROM orders
		WHERE order_date >= date_trunc('year', now())
		GROUP BY customer_id, best_month
		) AS subquery
	ORDER BY subquery.customer_id ASC, subquery.qty DESC;

-- самый популярный заказываемый продукт за последний месяц
CREATE OR REPLACE VIEW best_product_view AS
	SELECT DISTINCT ON (subquery.customer_id) subquery.customer_id, subquery.product_id AS best_product FROM (
		SELECT orders.customer_id, order_items.product_id, SUM(order_items.quantity) AS qty FROM orders
		JOIN order_items
		ON orders.order_id = order_items.order_id
		WHERE orders.order_date >= date_trunc('month', now())
		GROUP BY orders.customer_id, order_items.product_id
		) AS subquery
	ORDER BY subquery.customer_id ASC, subquery.qty DESC;

-- кол-во заказов, где стоимость заказа превышает среднюю стоимость всех его заказов за последний год
CREATE OR REPLACE VIEW more_avg_qty_view AS
	SELECT subquery.customer_id, COUNT(*) AS more_avg_qty FROM (
		SELECT orders.customer_id, orders.order_id, (order_items.unit_price * order_items.quantity) AS order_price, 
		CAST(AVG(order_items.unit_price * order_items.quantity) OVER(PARTITION BY orders.customer_id) AS decimal(6, 2)) AS avg_price FROM orders
		JOIN order_items
		ON orders.order_id = order_items.order_id
		WHERE orders.order_date >= date_trunc('year', now())
		) AS subquery
	WHERE subquery.order_price > subquery.avg_price
	GROUP BY subquery.customer_id;


--объединим запросы
SELECT DISTINCT customers.customer_id, customers.cust_first_name, customers.cust_last_name, employees_info_view.first_name AS empl_first_name, employees_info_view.last_name AS empl_last_name, employees_info_view.job_title, employees_info_view.department_name, employees_info_view.region_name, employees_info_view.country_name, employees_info_view.manager_first_name, employees_info_view.manager_last_name, last_30_days_view.last_30_days, last_3_month_view.last_3_month, best_month_view.best_month, best_product_view.best_product, more_avg_qty_view.more_avg_qty FROM customers

LEFT JOIN employees_info_view ON customers.account_mgr_id = employees_info_view.employee_id
LEFT JOIN last_30_days_view ON customers.customer_id = last_30_days_view.customer_id
LEFT JOIN last_3_month_view ON customers.customer_id = last_3_month_view.customer_id
LEFT JOIN best_month_view ON customers.customer_id = best_month_view.customer_id
LEFT JOIN best_product_view ON customers.customer_id = best_product_view.customer_id
LEFT JOIN more_avg_qty_view ON customers.customer_id = more_avg_qty_view.customer_id
	
ORDER BY customers.customer_id ASC
```