-- script that ranks country origins of bands, ordered by the number of (non-unique) fans
SELECT
    SUBSTRING_INDEX(origin, ',', 1) AS origin,
    SUM(fans) AS nb_fans
FROM
    metal_bands
GROUP BY
    origin
ORDER BY
    nb_fans DESC;